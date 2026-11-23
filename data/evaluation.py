"""Evaluation, ranking-distortion and statistical-testing helpers."""
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold


def cm_metrics(y, yhat):
    """All threshold metrics from one confusion matrix (zero_division -> 0)."""
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    rec = tp / (tp + fn) if tp + fn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return dict(recall=rec, precision=prec, f1=f1, gmean=float(np.sqrt(rec * spec)),
                mcc=float(mcc), accuracy=(tp + tn) / len(y))


def scores(model, X):
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") \
        else model.decision_function(X)


def evaluate_cell(make_model, X, y, repeats=5, k=5, seed=42):
    """Repeated stratified k-fold. Returns pooled-per-repeat and fold-level tables."""
    pooled, folds = [], []
    for r in range(repeats):
        skf = StratifiedKFold(k, shuffle=True, random_state=seed + r)
        oof = np.zeros(len(y), int)
        aucs = []
        for tr, te in skf.split(X, y):
            m = make_model(seed + r).fit(X.iloc[tr], y[tr])
            oof[te] = m.predict(X.iloc[te])
            folds.append(dict(repeat=r, **cm_metrics(y[te], oof[te]),
                              auc=roc_auc_score(y[te], scores(m, X.iloc[te])),
                              ap=average_precision_score(y[te], scores(m, X.iloc[te]))))
        p = cm_metrics(y, oof)                # Forman-Scholz pooled estimator
        p["auc"] = float(np.mean([f["auc"] for f in folds[-k:]]))
        pooled.append(p)
    return pooled, folds


def distortion(acc, metric):
    """Accuracy-versus-metric ranking distortion for one fault (arrays over classifiers)."""
    acc, metric = np.asarray(acc), np.asarray(metric)
    tau = stats.kendalltau(acc, metric).statistic
    r_a = stats.rankdata(-acc, method="average")
    r_m = stats.rankdata(-metric, method="average")
    disp = float(np.mean(np.abs(r_a - r_m)))
    dsr = (metric.max() - metric.min()) / max(acc.max() - acc.min(), 1e-6)
    top_agree = bool(np.argmax(acc) == np.argmax(metric))
    return dict(tau=float(tau), displacement=disp, dsr=float(dsr), top1_agrees=top_agree)


def nadeau_bengio(d, n_train, n_test):
    """Corrected resampled t-test on paired fold-level differences d."""
    d = np.asarray(d, float)
    J, var = len(d), d.var(ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / J + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), J - 1))


def bayes_correlated_t(d, n_train, n_test, rope=0.01):
    """Posterior P(left), P(rope), P(right) for the mean paired difference."""
    d = np.asarray(d, float)
    J, var = len(d), d.var(ddof=1)
    scale = np.sqrt((1 / J + n_test / n_train) * max(var, 1e-12))
    post = stats.t(J - 1, loc=d.mean(), scale=scale)
    left, right = post.cdf(-rope), post.sf(rope)
    return dict(p_left=float(left), p_rope=float(1 - left - right), p_right=float(right))


Q_ALPHA_05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}

def friedman_nemenyi(M):
    """M: blocks x classifiers matrix of a metric. Returns chi2, p, mean ranks, CD."""
    N, k = M.shape
    chi2, p = stats.friedmanchisquare(*[M[:, j] for j in range(k)])
    ranks = np.vstack([stats.rankdata(-row) for row in M]).mean(axis=0)
    cd = Q_ALPHA_05[k] * np.sqrt(k * (k + 1) / (6.0 * N))
    return float(chi2), float(p), ranks, float(cd)
