"""
core.py: all computation for the Rare Fault Lab app. No Streamlit imports here, so it can be tested alone.

Sections
  1. constants and result loaders (the thesis CSV files in ./data)
  2. statistics (distortion, Friedman/Nemenyi, Nadeau-Bengio, Bayesian correlated t, recommendation rule)
  3. AI4I 2020: loading, documented failure rules, model construction, training bundle
  4. CWRU: signal windowing, features, task construction, grouped folds, training bundle
"""
import io, os, re, warnings, zipfile
import numpy as np
import pandas as pd
from scipy import stats, signal
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

ORDER = ["LR", "DT", "SVM", "RF", "LGBM", "XGB"]          # parsimony order, simplest first
FULL = {"LR": "Logistic Regression", "DT": "Decision Tree", "SVM": "Support Vector Machine",
        "RF": "Random Forest", "LGBM": "LightGBM", "XGB": "XGBoost"}
FAULTS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FAULT_NAME = {"TWF": "Tool wear failure", "HDF": "Heat dissipation failure", "PWF": "Power failure",
              "OSF": "Overstrain failure", "RNF": "Random failure"}
METRICS = {"mcc": "MCC", "f1": "F1", "recall": "Recall", "precision": "Precision",
           "gmean": "G-Mean", "auc": "ROC-AUC", "accuracy": "Accuracy"}
NUM = ["air_temp", "proc_temp", "rot_speed", "torque", "tool_wear"]
AI4I_FEATURES = NUM + ["type"]
CWRU_FEATS = ["mean", "std", "rms", "peak", "p2p", "skew", "kurt", "crest", "shape", "impulse"]
CWRU_FAULT = {"B": "Ball", "IR": "Inner race", "OR": "Outer race"}
MCC_FLOOR = 0.10
W, HOP, TARGET_FS = 2048, 1024, 12_000


# ---------------------------------------------------------------- 1. loaders
def _csv(name):
    p = os.path.join(DATA, name)
    return pd.read_csv(p) if os.path.exists(p) else None


def load_p1(prefix="phase1"):
    return dict(summary=_csv(f"{prefix}_summary.csv"), pooled=_csv(f"{prefix}_pooled.csv"),
                folds=_csv(f"{prefix}_folds.csv"), distortion=_csv(f"{prefix}_distortion.csv"),
                friedman=_csv(f"{prefix}_friedman.csv"), recommendation=_csv(f"{prefix}_recommendation.csv"))


def load_p2():
    parts = [pd.read_csv(os.path.join(DATA, f)) for f in sorted(os.listdir(DATA))
             if re.match(r"phase2_.*pct\.csv$", f)]
    if not parts:
        return None
    d = pd.concat(parts, ignore_index=True).rename(columns={"fault": "ftype"})
    d["cond"] = d.ftype + "@" + (d.prevalence * 100).round().astype(int).astype(str) + "%"
    return d


def load_windows():
    return _csv("cwru_windows.csv")


def cond_order(p2):
    return list(p2.drop_duplicates("cond").sort_values(["ftype", "prevalence"]).cond)


def mean_table(df, metric, key):
    """classifier x key table of means (key = 'fault' for Phase 1, 'cond' for Phase 2)."""
    t = df.groupby(["classifier", key])[metric].mean().unstack(key)
    return t.reindex([c for c in ORDER if c in t.index])


def sd_table(df, metric, key):
    t = df.groupby(["classifier", key])[metric].std().unstack(key)
    return t.reindex([c for c in ORDER if c in t.index])


# ---------------------------------------------------------------- 2. statistics
def distortion(acc, metric):
    acc, metric = np.asarray(acc, float), np.asarray(metric, float)
    tau = stats.kendalltau(acc, metric).statistic
    r_a, r_m = stats.rankdata(-acc), stats.rankdata(-metric)
    return dict(tau=float(tau), displacement=float(np.mean(np.abs(r_a - r_m))),
                dsr=float((metric.max() - metric.min()) / max(acc.max() - acc.min(), 1e-6)),
                top1_agrees=bool(np.argmax(acc) == np.argmax(metric)))


def nadeau_bengio(d, n_train, n_test):
    d = np.asarray(d, float); J, var = len(d), d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def bayes_correlated_t(d, n_train, n_test, rope=0.01):
    d = np.asarray(d, float); J, var = len(d), d.var(ddof=1)
    scale = np.sqrt((1 / J + n_test / n_train) * max(var, 1e-12))
    post = stats.t(J - 1, loc=d.mean(), scale=scale)
    left, right = post.cdf(-rope), post.sf(rope)
    return dict(p_left=float(left), p_rope=float(1 - left - right), p_right=float(right))


Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}


def friedman_nemenyi(M):
    """M: blocks x classifiers. Returns chi2, p, mean ranks, Nemenyi critical difference."""
    N, k = M.shape
    chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    ranks = np.vstack([stats.rankdata(-r) for r in M]).mean(axis=0)
    return float(chi2), float(p), ranks, float(Q05[k] * np.sqrt(k * (k + 1) / (6.0 * N)))


def pairwise_vs_best(folds, fault, rope=0.01, n_train=8000, n_test=2000, seed=0):
    """Each classifier against the best-MCC classifier for one fault (fold-level, thesis rule)."""
    fo = folds[folds.fault == fault]
    piv = fo.pivot_table(index=["repeat", "fold"], columns="classifier", values="mcc")
    piv = piv[[c for c in ORDER if c in piv.columns]]
    best = piv.mean().idxmax(); rng = np.random.default_rng(seed); rows = []
    for c in piv.columns:
        d = (piv[c] - piv[best]).to_numpy()
        bs = rng.choice(piv[c].to_numpy(), (2000, len(piv)), replace=True).mean(axis=1)
        pr = bayes_correlated_t(d, n_train, n_test, rope) if c != best else dict(p_left=0., p_rope=1., p_right=0.)
        t, p = nadeau_bengio(d, n_train, n_test) if c != best else (0.0, 1.0)
        rows.append(dict(classifier=c, mean_mcc=piv[c].mean(), sd=piv[c].std(), ci_low=np.percentile(bs, 2.5),
                         ci_high=np.percentile(bs, 97.5), diff_vs_best=d.mean(), nb_t=t, nb_p=p,
                         p_worse=pr["p_left"], p_equivalent=pr["p_rope"], p_better=pr["p_right"],
                         no_worse=(c == best) or (pr["p_rope"] + pr["p_right"]) >= 0.95))
    return best, pd.DataFrame(rows)


def thesis_recommendation(folds, floor=MCC_FLOOR, rope=0.01, n_train=8000, n_test=2000):
    """Parsimony rule with the minimum-MCC floor. Returns one row per fault."""
    out = []
    for f in FAULTS:
        best, t = pairwise_vs_best(folds, f, rope, n_train, n_test)
        ok = list(t[t.no_worse].classifier)
        zero_all = not (t.ci_low > 0).any()
        best_mcc = float(t.loc[t.classifier == best, "mean_mcc"].iloc[0])
        if zero_all:
            pick = "none (every MCC interval includes 0)"
        elif best_mcc < floor:
            pick = f"none (best MCC below floor {floor:g})"
        else:
            pick = next(c for c in ORDER if c in ok)
        out.append(dict(fault=f, best_by_mcc=best, best_mcc=round(best_mcc, 4),
                        acceptable=" ".join(c for c in ORDER if c in ok), recommended=pick))
    return pd.DataFrame(out)


def tolerance_pick(means, tol, floor=MCC_FLOOR):
    """Quick rule: simplest classifier within `tol` of the best mean MCC; None under the floor."""
    s = means.dropna(); best = s.max()
    if best < floor:
        return None, s[s >= best - tol].index.tolist()
    ok = [c for c in ORDER if c in s.index and s[c] >= best - tol - 1e-9]
    return ok[0], ok


def confusion_metrics(tp, fp, fn, tn):
    n = tp + fp + fn + tn
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return dict(recall=rec, precision=prec, f1=f1, gmean=float(np.sqrt(rec * spec)), mcc=float(mcc),
                accuracy=(tp + tn) / n if n else 0.0)


def scenario(prev, recall, spec, n=10_000):
    P = int(round(n * prev)); Ng = n - P
    tp = int(round(P * recall)); fn = P - tp; tn = int(round(Ng * spec)); fp = Ng - tn
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, **confusion_metrics(tp, fp, fn, tn))


def champion_table(summary, floor=MCC_FLOOR):
    """Best classifier per fault by mean pooled MCC, with a reliability label."""
    rows = []
    for f in FAULTS:
        s = summary[summary.fault == f].sort_values("mcc", ascending=False)
        r = s.iloc[0]
        label = "Reliable" if r.mcc >= 0.5 else ("Weak" if r.mcc >= floor else "Not reliable")
        rows.append(dict(fault=f, champion=r.classifier, mcc=r.mcc, recall=r.recall, precision=r.precision,
                         reliability=label))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3. AI4I 2020
def _clean_ai4i(df):
    df = df.rename(columns={"Air temperature [K]": "air_temp", "Process temperature [K]": "proc_temp",
                            "Rotational speed [rpm]": "rot_speed", "Torque [Nm]": "torque",
                            "Tool wear [min]": "tool_wear", "Type": "type"})
    if not pd.api.types.is_numeric_dtype(df["type"]):          # object (pandas 2) or string (pandas 3) dtype
        df["type"] = df["type"].astype(str).str.strip().str.upper().map({"L": 0, "M": 1, "H": 2})
    if df["type"].isna().any():
        raise ValueError("Type must be L, M or H")
    df["type"] = df["type"].astype(int)
    return df


def read_ai4i(src):
    """src: path, bytes (csv or zip) or file-like. Returns a cleaned frame with labels, or raises."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    if hasattr(src, "read"):
        raw = src.read(); src = io.BytesIO(raw)
        if raw[:2] == b"PK":
            z = zipfile.ZipFile(src)
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            src = io.BytesIO(z.read(name))
    df = _clean_ai4i(pd.read_csv(src))
    need = AI4I_FEATURES + FAULTS
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns after renaming: {miss}")
    if df[AI4I_FEATURES].isna().any().any():
        raise ValueError("Feature columns contain missing values")
    return df


UCI_URLS = ["https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"]


def find_ai4i():
    """Local copy first (data/ or the app folder), then the UCI archive. Returns (df, source) or (None, msg)."""
    for p in [os.path.join(DATA, "ai4i2020.csv"), os.path.join(HERE, "ai4i2020.csv")]:
        if os.path.exists(p):
            return read_ai4i(p), p
    try:
        import urllib.request
        for u in UCI_URLS:
            with urllib.request.urlopen(u, timeout=30) as r:
                return read_ai4i(r.read()), u
    except Exception as e:                                   # network blocked, URL moved, bad file
        return None, f"download failed ({type(e).__name__}: {e})"
    return None, "no source worked"


def power_w(torque, rpm):
    return np.asarray(torque, float) * np.asarray(rpm, float) * 2 * np.pi / 60


def rule_flags(df):
    """Failure conditions as documented for the AI4I 2020 data set (Matzka, 2020). Boolean columns."""
    lim = pd.Series(df["type"]).map({0: 11000, 1: 12000, 2: 13000}).to_numpy()
    pw = power_w(df["torque"], df["rot_speed"])
    return pd.DataFrame({
        "TWF": (df["tool_wear"] >= 200) & (df["tool_wear"] <= 240),       # random replacement window
        "HDF": ((df["proc_temp"] - df["air_temp"]) < 8.6) & (df["rot_speed"] < 1380),
        "PWF": (pw < 3500) | (pw > 9000),
        "OSF": (df["tool_wear"] * df["torque"]) > lim,
        "RNF": pd.Series(False, index=df.index)}, index=df.index)


def rule_audit(df):
    """How well each documented rule reproduces the labels (flag = fault)."""
    fl = rule_flags(df); rows = []
    for f in FAULTS:
        y = df[f].to_numpy(int); p = fl[f].to_numpy(int)
        tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
        fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
        rows.append(dict(fault=f, faults=int(y.sum()), flagged=int(p.sum()), **confusion_metrics(tp, fp, fn, tn)))
    return pd.DataFrame(rows)


class _Adaptive:
    """Factory so that imblearn is only imported when available."""
    @staticmethod
    def build(cat_idx, seed):
        from imblearn.over_sampling import SMOTENC

        class AdaptiveSMOTENC(SMOTENC):                     # neighbourhood shrinks with the minority count
            def __init__(self, categorical_features, k_max=5, sampling_strategy=1.0, random_state=None):
                super().__init__(categorical_features=categorical_features, sampling_strategy=sampling_strategy,
                                 random_state=random_state, k_neighbors=k_max)
                self.k_max = k_max

            def _fit_resample(self, X, y):
                n_min = int(np.sum(y == 1))
                if n_min < 2:
                    return X, y
                self.k_neighbors = min(self.k_max, n_min - 1)
                return super()._fit_resample(X, y)
        return AdaptiveSMOTENC(cat_idx, random_state=seed)


def have(mod):
    try:
        __import__(mod); return True
    except Exception:
        return False


def capabilities():
    return dict(imblearn=have("imblearn"), xgboost=have("xgboost"), lightgbm=have("lightgbm"))


def available_classifiers():
    cap = capabilities()
    return [c for c in ORDER if (c != "XGB" or cap["xgboost"]) and (c != "LGBM" or cap["lightgbm"])]


def make_classifier(name, seed=0, balanced=False, y=None):
    cw = "balanced" if balanced else None
    if name == "LR":
        return LogisticRegression(max_iter=1000, class_weight=cw)
    if name == "DT":
        return DecisionTreeClassifier(max_depth=10, class_weight=cw, random_state=seed)
    if name == "SVM":
        return SVC(probability=True, class_weight=cw, random_state=seed)
    if name == "RF":
        return RandomForestClassifier(100, random_state=seed, n_jobs=1, class_weight=cw)
    if name == "LGBM":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=100, random_state=seed, verbose=-1, n_jobs=1, class_weight=cw)
    if name == "XGB":
        from xgboost import XGBClassifier
        spw = 1.0
        if balanced and y is not None:
            n_pos = max(float(np.sum(y == 1)), 1.0); spw = (len(y) - n_pos) / n_pos
        return XGBClassifier(n_estimators=100, learning_rate=.1, max_depth=6, random_state=seed,
                             eval_metric="logloss", n_jobs=1, scale_pos_weight=spw)
    raise KeyError(name)


def ai4i_pipeline(name, seed=0, smote=True, y=None):
    """Thesis pipeline: scale numerics, SMOTE-NC to parity (training data only), one-hot type, classifier.
    Without imbalanced-learn the same pipeline is built with class weights instead of resampling."""
    if smote and capabilities()["imblearn"]:
        from imblearn.pipeline import Pipeline as ImbPipeline
        scale = ColumnTransformer([("num", StandardScaler(), list(range(5))), ("cat", "passthrough", [5])])
        onehot = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore", categories=[[0, 1, 2]]), [5])],
                                   remainder="passthrough")
        return ImbPipeline([("scale", scale), ("smote", _Adaptive.build([5], seed)),
                            ("encode", onehot), ("clf", make_classifier(name, seed))]), True
    prep = ColumnTransformer([("num", StandardScaler(), list(range(5))),
                              ("cat", OneHotEncoder(handle_unknown="ignore", categories=[[0, 1, 2]]), [5])])
    return SkPipeline([("prep", prep), ("clf", make_classifier(name, seed, True, y))]), False


def ai4i_feature_names(used_smote):
    t = ["type_L", "type_M", "type_H"]
    return (t + NUM) if used_smote else (NUM + t)


def _calibrator(p_oof, y):
    """Platt scaling on out-of-fold scores: turns the resampled model's score into a real-prevalence probability."""
    z = np.log(np.clip(p_oof, 1e-6, 1 - 1e-6) / (1 - np.clip(p_oof, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    if y.sum() < 3 or (1 - y).sum() < 3:
        return None
    return LogisticRegression(C=1e6, max_iter=1000).fit(z, y)


def _apply_cal(cal, p):
    if cal is None:
        return np.asarray(p)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return cal.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


def best_threshold(y, p):
    """Threshold on the raw score that maximises MCC out of fold. When many thresholds tie (a clean
    separation), the middle of the tied range is used, not its edge."""
    ts = np.unique(np.r_[np.linspace(0.02, 0.98, 49), 0.5]); ms = []
    for t in ts:
        tn, fp, fn, tp = confusion_matrix(y, (p >= t).astype(int), labels=[0, 1]).ravel()
        ms.append(confusion_metrics(tp, fp, fn, tn)["mcc"])
    ms = np.array(ms); top = ts[ms >= ms.max() - 1e-9]
    return float(np.median(top)), float(ms.max())


def threshold_sweep(y, p):
    rows = []
    for t in np.linspace(0.02, 0.98, 49):
        yh = (p >= t).astype(int); tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
        rows.append(dict(threshold=t, tp=tp, fp=fp, fn=fn, tn=tn, **confusion_metrics(tp, fp, fn, tn)))
    return pd.DataFrame(rows)


def _importance(model, X, y, names_fn):
    rng = np.random.default_rng(0); pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    take = np.r_[pos[:300], rng.choice(neg, min(len(neg), 2500), replace=False)]
    r = permutation_importance(model, X.iloc[take], y[take], scoring="average_precision",
                               n_repeats=3, random_state=0, n_jobs=1)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def train_ai4i(df, fault, name, seed=0, smote=True):
    """Fit on all rows; out-of-fold (5-fold) scores give calibration, threshold and honest accuracy."""
    X, y = df[AI4I_FEATURES].reset_index(drop=True), df[fault].to_numpy(int)
    skf = StratifiedKFold(5, shuffle=True, random_state=42 + seed)
    oof = np.zeros(len(y)); used = False
    for tr, te in skf.split(X, y):
        m, used = ai4i_pipeline(name, seed, smote, y[tr])
        m.fit(X.iloc[tr], y[tr]); oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    model, used = ai4i_pipeline(name, seed, smote, y)
    model.fit(X, y)
    thr, m_opt = best_threshold(y, oof)
    tn, fp, fn, tp = confusion_matrix(y, (oof >= thr).astype(int), labels=[0, 1]).ravel()
    tn5, fp5, fn5, tp5 = confusion_matrix(y, (oof >= 0.5).astype(int), labels=[0, 1]).ravel()
    return dict(model=model, task=fault, classifier=name, smote=used, y=y, oof=oof, cal=_calibrator(oof, y),
                threshold=thr, at_opt=confusion_metrics(tp, fp, fn, tn), at_half=confusion_metrics(tp5, fp5, fn5, tn5),
                auc=float(roc_auc_score(y, oof)) if y.sum() else np.nan,
                ap=float(average_precision_score(y, oof)) if y.sum() else np.nan,
                importance=_importance(model, X, y, None), n=len(y), n_pos=int(y.sum()),
                features=AI4I_FEATURES, lo=X.min(), hi=X.max())


def predict_with(bundle, X, threshold=None):
    """Raw score, calibrated probability and alarm for rows of X (same columns as training)."""
    thr = bundle["threshold"] if threshold is None else threshold
    p = bundle["model"].predict_proba(X[bundle["features"]])[:, 1]
    return pd.DataFrame({"score": p, "probability": _apply_cal(bundle["cal"], p), "alarm": (p >= thr).astype(int)},
                        index=X.index)


def reliability(mcc, floor=MCC_FLOOR):
    return "Reliable" if mcc >= 0.5 else ("Weak" if mcc >= floor else "Not reliable")


# ---------------------------------------------------------------- 4. CWRU
def features_of(w):
    w = np.asarray(w, float)
    mu, sd = w.mean(), w.std(); rms = np.sqrt(np.mean(w ** 2)); ab = np.abs(w); mabs = ab.mean(); pk = ab.max()
    return [mu, sd, rms, pk, w.max() - w.min(), stats.skew(w), stats.kurtosis(w, fisher=False),
            pk / rms, rms / mabs, pk / mabs]


def to_target_rate(x, fs):
    if fs == TARGET_FS:
        return np.asarray(x, float)
    if fs % TARGET_FS:
        raise ValueError(f"Cannot decimate {fs} Hz to {TARGET_FS} Hz by an integer factor; resample first.")
    return signal.decimate(np.asarray(x, float), fs // TARGET_FS, ftype="fir", zero_phase=True)


def signal_to_features(x, fs=TARGET_FS):
    x = to_target_rate(x, fs)
    if len(x) < W:
        raise ValueError(f"Need at least {W} samples at {TARGET_FS} Hz, got {len(x)}.")
    rows = [features_of(x[s:s + W]) for s in range(0, len(x) - W + 1, HOP)]
    return pd.DataFrame(rows, columns=CWRU_FEATS), x


def read_signal(file, name):
    """Accept .csv/.txt (one numeric column, or the column you choose later) and .mat (DE_time)."""
    raw = file.read() if hasattr(file, "read") else file
    if name.lower().endswith(".mat"):
        from scipy import io as sio
        m = sio.loadmat(io.BytesIO(raw))
        keys = [k for k in m if k.endswith("DE_time")]
        if not keys:
            raise ValueError("No *_DE_time channel in this .mat file.")
        return pd.DataFrame({"DE_time": np.ravel(m[keys[0]]).astype(float)})
    df = pd.read_csv(io.BytesIO(raw), header=0)
    num = df.select_dtypes("number")
    if num.shape[1] == 0:                                   # file without a header
        num = pd.read_csv(io.BytesIO(raw), header=None).select_dtypes("number")
    return num


def make_task(df, fault, prev, seed=0):
    """All healthy windows plus fault windows thinned (evenly across records) to prevalence `prev`."""
    h = df[df.fault_type == "Normal"]; f = df[df.fault_type == fault]
    if prev is None:
        return pd.concat([h, f], ignore_index=True)
    n_f = int(round(prev / (1 - prev) * len(h))); rng = np.random.default_rng(seed)
    files = sorted(f.file.unique()); quota = np.full(len(files), n_f // len(files)); quota[: n_f % len(files)] += 1
    parts = [f[f.file == fl].iloc[rng.choice(len(f[f.file == fl]), min(q, len(f[f.file == fl])), replace=False)]
             for fl, q in zip(files, quota)]
    return pd.concat([h] + parts, ignore_index=True)


def grouped_folds(d, seed=0, k=4):
    """Fold i holds one healthy record; fault records are spread round-robin. No record is split across folds."""
    rng = np.random.default_rng(seed)
    hf = sorted(d[d.label == 0].file.unique()); ff = sorted(d[d.label == 1].file.unique())
    k = min(k, len(hf)); rng.shuffle(hf); rng.shuffle(ff)
    fold_of = {f: i % k for i, f in enumerate(hf)}
    for j, f in enumerate(ff):
        fold_of[f] = j % k
    return d.file.map(fold_of).to_numpy(), k


def cwru_pipeline(name, seed=0, smote=True, y=None):
    clf = make_classifier(name, seed, balanced=not (smote and capabilities()["imblearn"]), y=y)
    if smote and capabilities()["imblearn"]:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        return ImbPipeline([("z", StandardScaler()), ("smote", SMOTE(k_neighbors=5, random_state=seed)),
                            ("clf", clf)]), True
    return SkPipeline([("z", StandardScaler()), ("clf", clf)]), False


def train_cwru(win, ftype, name, prev, seed=0, smote=True):
    d = make_task(win, ftype, prev, seed).reset_index(drop=True)
    X, y = d[CWRU_FEATS], d.label.to_numpy(int)
    fold, k = grouped_folds(d, seed)
    oof = np.zeros(len(y)); used = False
    for i in range(k):
        te = fold == i
        if te.sum() == 0 or y[te].sum() == 0 or y[~te].sum() == 0:
            oof[te] = np.nan; continue
        m, used = cwru_pipeline(name, seed, smote, y[~te]); m.fit(X[~te], y[~te])
        oof[te] = m.predict_proba(X[te])[:, 1]
    ok = ~np.isnan(oof)
    model, used = cwru_pipeline(name, seed, smote, y); model.fit(X, y)
    yy, pp = y[ok], oof[ok]
    thr, _ = best_threshold(yy, pp)
    tn, fp, fn, tp = confusion_matrix(yy, (pp >= thr).astype(int), labels=[0, 1]).ravel()
    tn5, fp5, fn5, tp5 = confusion_matrix(yy, (pp >= 0.5).astype(int), labels=[0, 1]).ravel()
    return dict(model=model, task=ftype, classifier=name, smote=used, prev=prev, y=yy, oof=pp,
                cal=_calibrator(pp, yy), threshold=thr, at_opt=confusion_metrics(tp, fp, fn, tn),
                at_half=confusion_metrics(tp5, fp5, fn5, tn5),
                auc=float(roc_auc_score(yy, pp)) if yy.sum() else np.nan, ap=float(average_precision_score(yy, pp)),
                importance=_importance(model, X, y, None), n=len(y), n_pos=int(y.sum()), features=CWRU_FEATS,
                lo=X.min(), hi=X.max())


def cwru_default_classifier(p2, ftype, prev):
    """Champion by mean MCC for the matching Phase 2 condition (falls back to the fault-type average)."""
    if p2 is None:
        return "RF"
    cand = p2[(p2.ftype == ftype) & (np.isclose(p2.prevalence, prev if prev else 0.05))]
    if cand.empty:
        cand = p2[p2.ftype == ftype]
    if cand.empty:
        cand = p2
    m = cand.groupby("classifier").mcc.mean()
    m = m[[c for c in m.index if c in available_classifiers()]]
    return m.idxmax() if len(m) else "RF"


def outside_range(bundle, X, margin=0.25):
    """Share of input features outside the training range (widened by `margin`) per row. Predictions for rows
    far outside what the model saw are extrapolations and should not be trusted."""
    lo, hi = bundle["lo"], bundle["hi"]; span = (hi - lo).replace(0, 1)
    lo, hi = lo - margin * span, hi + margin * span
    Xf = X[bundle["features"]]
    return ((Xf < lo) | (Xf > hi)).mean(axis=1)
