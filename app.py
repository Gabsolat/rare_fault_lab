"""
Rare Fault Lab (Streamlit). Rare-fault classifier comparison on AI4I 2020 (Phase 1) and CWRU bearings (Phase 2),
with live prediction on user-supplied data.   Run:  streamlit run app.py
"""
import io, time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core as C

st.set_page_config(page_title="Rare Fault Lab", page_icon=None, layout="wide")

CLR = {"LR": "#1f77b4", "DT": "#2ca02c", "SVM": "#ff7f0e", "RF": "#9467bd", "XGB": "#d62728", "LGBM": "#17becf"}
POS = {"TWF": 46, "HDF": 115, "PWF": 95, "OSF": 98, "RNF": 19}     # AI4I 2020 positives, N = 10,000
PAGES = ["Overview", "Phase 1 · AI4I results", "Phase 2 · CWRU results", "Accuracy paradox",
         "Statistics & recommendation", "Signal features", "Predict · AI4I machine", "Predict · CWRU bearing"]


# ------------------------------------------------------------------ cached loaders
@st.cache_data(show_spinner=False)
def p1_data(prefix="phase1"):
    return C.load_p1(prefix)


@st.cache_data(show_spinner=False)
def p2_data():
    return C.load_p2()


@st.cache_data(show_spinner=False)
def windows():
    return C.load_windows()


@st.cache_data(show_spinner=False)
def recommendation(floor):
    return C.thesis_recommendation(p1_data()["folds"], floor)


@st.cache_data(show_spinner=False)
def pairwise(fault):
    return C.pairwise_vs_best(p1_data()["folds"], fault)


@st.cache_data(show_spinner=False)
def find_ai4i():
    return C.find_ai4i()


@st.cache_resource(show_spinner=False)
def ai4i_bundle(fault, clf, smote, data_id):
    return C.train_ai4i(st.session_state["_ai4i_df"], fault, clf, smote=smote)


@st.cache_resource(show_spinner=False)
def cwru_bundle(ftype, clf, prev, smote):
    return C.train_cwru(windows(), ftype, clf, prev, smote=smote)


def ai4i_df():
    """Raw AI4I frame from disk, the UCI archive, or the sidebar upload."""
    if "_ai4i_df" in st.session_state:
        return st.session_state["_ai4i_df"], st.session_state.get("_ai4i_src", "uploaded")
    df, src = find_ai4i()
    if df is not None:
        st.session_state["_ai4i_df"], st.session_state["_ai4i_src"] = df, src
        return df, src
    return None, src


# ------------------------------------------------------------------ figures
def fig_heat(tab, title, fmt="{:.2f}", vmax=None):
    fig, ax = plt.subplots(figsize=(7.2, 0.55 * len(tab) + 1.4))
    v = tab.to_numpy(float); vmax = vmax or max(np.nanmax(v), 1e-9)
    im = ax.imshow(v, cmap="Blues", vmin=min(0, np.nanmin(v)), vmax=vmax, aspect="auto")
    ax.set_xticks(range(tab.shape[1]), tab.columns); ax.set_yticks(range(tab.shape[0]), tab.index)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            ax.text(j, i, fmt.format(v[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if v[i, j] > 0.55 * vmax else "black")
    ax.set_title(title, loc="left", fontsize=11); fig.colorbar(im, ax=ax, shrink=.85); fig.tight_layout()
    return fig


def fig_bump(tab, title):
    ranks = tab.rank(ascending=False, method="min"); fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(tab.shape[1]); jit = {}
    for k in range(tab.shape[1]):
        g = ranks.iloc[:, k]
        for r, idx in g.groupby(g).groups.items():
            for n, c in enumerate(idx):
                jit[(c, k)] = (n - (len(idx) - 1) / 2) * 0.07
    for c in tab.index:
        y = [ranks.loc[c].iloc[k] + jit[(c, k)] for k in range(tab.shape[1])]
        ax.plot(x, y, "-o", color=CLR[c], label=c, lw=2, ms=5)
        ax.text(x[-1] + 0.08, y[-1], c, va="center", fontsize=9, color=CLR[c])
    ax.set_xticks(x, tab.columns); ax.set_yticks(range(1, len(tab) + 1)); ax.invert_yaxis()
    ax.set_ylabel("Rank (1 = best)"); ax.set_title(title, loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False); ax.set_xlim(-0.2, x[-1] + 0.6); fig.tight_layout()
    return fig


def fig_box(df, col, by, title, order=None):
    order = order or [c for c in C.ORDER if c in set(df[by])]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bp = ax.boxplot([df[df[by] == c][col] for c in order], patch_artist=True, showmeans=True); ax.set_xticklabels(order)
    for p, c in zip(bp["boxes"], order):
        p.set_facecolor(CLR.get(c, "#888")); p.set_alpha(.55)
    ax.set_title(title, loc="left", fontsize=11); ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylabel(col.upper()); fig.tight_layout()
    return fig


def fig_cd(ranks, names, cd, title):
    order = np.argsort(ranks); fig, ax = plt.subplots(figsize=(7.2, 2.6))
    for i, j in enumerate(order):
        ax.plot(ranks[j], 0, "o", color=CLR[names[j]], ms=9)
        ax.annotate(names[j], (ranks[j], 0), xytext=(0, 12 if i % 2 == 0 else -18), textcoords="offset points",
                    ha="center", fontsize=9)
    lo = ranks.min(); ax.plot([lo, lo + cd], [0.35, 0.35], "k-", lw=3); ax.text(lo + cd / 2, 0.42, f"CD = {cd:.2f}", ha="center", fontsize=9)
    ax.set_ylim(-0.6, 0.8); ax.set_yticks([]); ax.set_xlabel("Mean rank (1 = best)")
    ax.set_title(title, loc="left", fontsize=11); ax.spines[["top", "right", "left"]].set_visible(False); fig.tight_layout()
    return fig


def fig_sweep(sw, thr, title):
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for m, c in [("recall", "#1f77b4"), ("precision", "#d62728"), ("mcc", "#2ca02c")]:
        ax.plot(sw.threshold, sw[m], label=C.METRICS[m], color=c, lw=2)
    ax.axvline(thr, color="k", ls="--", lw=1); ax.text(thr, 1.02, " chosen", fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_xlabel("Decision threshold on the model score"); ax.legend(frameon=False, ncol=3, loc="lower left")
    ax.set_title(title, loc="left", fontsize=11); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    return fig


def show(fig):
    st.pyplot(fig); plt.close(fig)


def pct(x):
    return f"{100 * x:.1f}%"


# ------------------------------------------------------------------ pages
def page_overview():
    p1, p2 = p1_data(), p2_data(); S = p1["summary"]
    st.title("Rare Fault Lab")
    st.markdown("**Machine learning classifier comparison for rare fault detection under class imbalance in "
                "manufacturing.** Every number in the results pages comes from the thesis experiments. "
                "The two prediction pages train the same classifiers on the same data and score data you enter.")
    st.subheader("Aim")
    st.write("To compare six classifiers (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM and SVM) "
             "on rare fault types in the AI4I 2020 data, under one controlled SMOTE baseline, using only metrics that "
             "respond to the minority class and are reported per fault type. The study finds the best classifier for "
             "each fault and measures how far accuracy misleads the choice. Phase 2 repeats the comparison on CWRU "
             "bearing vibration windows as an external check.")
    st.subheader("Objectives and where they appear")
    st.table(pd.DataFrame([
        ["O1", "Literature review of classifier comparisons on imbalanced maintenance data", "Thesis Chapter 2"],
        ["O2", "Imbalance profile of each fault type", "Phase 1 · AI4I results → Imbalance; Predict · AI4I machine → data audit"],
        ["O3", "Leakage-safe SMOTE inside the training folds", "Both prediction pages train with the same pipeline (SMOTE-NC or SMOTE inside each fold)"],
        ["O4", "Six classifiers × five faults = 30 cells", "Phase 1 · AI4I results"],
        ["O5", "Minority-class metrics, best classifier per fault and metric", "Phase 1 and Phase 2 results; Statistics & recommendation"],
        ["O6", "Tuning and robustness of the ranking", "Run phase1_tune.py; results appear under Statistics → Tuning when tuned_stability.csv is in data/"],
        ["O7", "Accuracy ranking versus minority-metric ranking", "Accuracy paradox"]],
        columns=["#", "Objective", "Page"]).set_index("#"))
    st.subheader("Champion classifier per fault (Phase 1, mean pooled MCC)")
    ch = C.champion_table(S); ch["name"] = ch.fault.map(C.FAULT_NAME)
    ch["base accuracy"] = [1 - POS[f] / 10000 for f in ch.fault]
    st.dataframe(ch[["fault", "name", "champion", "mcc", "recall", "precision", "reliability", "base accuracy"]]
                 .round(3), hide_index=True)
    n_below = int(sum(S[(S.fault == f)].accuracy.lt(1 - POS[f] / 10000).sum() for f in C.FAULTS))
    c1, c2, c3 = st.columns(3)
    c1.metric("Cells where accuracy is below the always-negative rule", f"{n_below} of {len(S)}")
    c2.metric("Best Phase 1 MCC", f"{S.mcc.max():.3f}")
    if p2 is not None:
        c3.metric("Best Phase 2 mean MCC (any classifier)", f"{C.mean_table(p2, 'mcc', 'cond').mean(axis=1).max():.3f}")
    st.info("Random failures (RNF) are not predictable from the sensor columns, and tool wear failures (TWF) only weakly. "
            "The prediction page says so for those two faults instead of reporting a confident number.")
    cap = C.capabilities()
    st.caption("Libraries on this server: " + ", ".join(f"{k} {'yes' if v else 'no'}" for k, v in cap.items()) +
               ". Without imbalanced-learn the prediction pages use class weights instead of SMOTE; "
               "without xgboost or lightgbm those two classifiers are not offered for live prediction.")


def page_phase(which):
    p1, p2 = p1_data(), p2_data()
    if which == 1:
        df, key, items, title = p1["pooled"], "fault", C.FAULTS, "AI4I 2020 · Phase 1"
        sub = ("Each column is one fault type predicted on its own: tool wear (TWF), heat dissipation (HDF), power (PWF), "
               "overstrain (OSF) and random (RNF). Pooled out-of-fold estimates, 5 repeats of stratified 5-fold CV, "
               "SMOTE-NC inside each training fold.")
    else:
        df, key, items, title = p2, "cond", C.cond_order(p2), "CWRU bearings · Phase 2"
        sub = ("Condition is fault type and prevalence, for example B@2% is ball faults thinned to 2% of windows. "
               "Grouped 4-fold CV (no record split across folds), 5 repeats.")
    st.title(title); st.caption(sub)
    tabs = st.tabs(["Heatmap and ranks", "Per-" + ("fault" if which == 1 else "condition") + " detail",
                    "Spread", "Alarms" if which == 1 else "Window data", "Full vs quick run" if which == 1 else "Phase 1 vs 2"])
    with tabs[0]:
        m = st.selectbox("Metric", list(C.METRICS), format_func=C.METRICS.get, key=f"m{which}")
        tab = C.mean_table(df, m, key)[items]
        c1, c2 = st.columns([1.05, 1])
        with c1:
            show(fig_heat(tab, f"Mean {C.METRICS[m]}", vmax=1.0 if m != "mcc" else None))
        with c2:
            show(fig_bump(tab, f"Rank by {C.METRICS[m]}"))
        avg = tab.mean(axis=1).rename("average"); st.dataframe(pd.concat([tab, avg], axis=1).round(3))
        if which == 1:
            st.markdown("**Always predicting 'no fault' scores accuracy** " +
                        ", ".join(f"{f} {1 - POS[f] / 10000:.3f}" for f in C.FAULTS) + " and MCC 0.")
    with tabs[1]:
        k = st.selectbox("Fault" if which == 1 else "Condition", items, key=f"k{which}",
                         format_func=(lambda f: f"{f} · {C.FAULT_NAME[f]}") if which == 1 else str)
        g = df[df[key] == k]
        t = g.groupby("classifier")[list(C.METRICS)].mean().reindex([c for c in C.ORDER if c in set(g.classifier)])
        sdv = g.groupby("classifier").mcc.std().reindex(t.index).rename("MCC sd (repeats)")
        st.dataframe(pd.concat([t, sdv], axis=1).round(3).rename(columns=C.METRICS))
        fig, ax = plt.subplots(figsize=(7.2, 3.2)); x = np.arange(len(t))
        for i, (m, c) in enumerate([("accuracy", "#aab4be"), ("recall", "#1f77b4"), ("precision", "#d62728"), ("mcc", "#2ca02c")]):
            ax.bar(x + (i - 1.5) * 0.2, t[m].clip(lower=0), 0.2, label=C.METRICS[m], color=c)
        ax.set_xticks(x, t.index); ax.set_ylim(0, 1.05); ax.legend(frameon=False, ncol=4, loc="lower left")
        ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); show(fig)
        if which == 1:
            best, pw = pairwise(k)
            st.markdown(f"**Fold-level comparison against the best classifier ({best}).** 'No worse' means the Bayesian "
                        "correlated t-test puts at least 95% probability on 'equivalent or better' (ROPE ±0.01).")
            st.dataframe(pw.round(3), hide_index=True)
    with tabs[2]:
        k = st.selectbox("Fault" if which == 1 else "Condition", items, key=f"sk{which}")
        if which == 1:
            show(fig_box(p1["folds"][p1["folds"].fault == k], "mcc", "classifier", f"Fold-level MCC, {k} (25 folds per classifier)"))
            st.caption("Each box holds the 25 fold-level MCC values (5 repeats × 5 folds). Wide boxes mean the ranking between close classifiers is not reliable.")
        else:
            show(fig_box(df[df.cond == k], "mcc", "classifier", f"MCC over the 5 repeats, {k}"))
            n_pos = int(df[df.cond == k].n_pos.iloc[0]); st.caption(f"Each repeat holds {n_pos} fault windows, so one missed window moves recall by {1 / n_pos:.3f}; gaps below about 0.05 are noise.")
    with tabs[3]:
        if which == 1:
            S = p1["summary"]; rows = []
            for f in C.FAULTS:
                for c in C.ORDER:
                    r = S[(S.fault == f) & (S.classifier == c)]
                    if r.empty: continue
                    r = r.iloc[0]; tp = r.recall * POS[f]; fp = tp * (1 - r.precision) / r.precision if r.precision > 0 else np.nan
                    rows.append(dict(fault=f, classifier=c, faults=POS[f], caught=round(tp), false_alarms=round(fp) if fp == fp else None,
                                     alarms_per_catch=round(fp / tp, 1) if tp > 0 and fp == fp else None))
            A = pd.DataFrame(rows)
            f = st.selectbox("Fault", C.FAULTS, key="alarm_f")
            st.dataframe(A[A.fault == f], hide_index=True)
            st.caption("Estimated per 10,000 rows from the mean recall and precision. The 'price' of detecting a rare fault is the false-alarm load: "
                       "precision below 0.5 means more false alarms than real faults.")
        else:
            w = windows()
            st.dataframe(w.groupby("fault_type").agg(records=("file", "nunique"), windows=("file", "size")))
            st.dataframe(w.groupby(["fault_type", "file"]).size().rename("windows").reset_index(), hide_index=True)
            st.caption("2048-sample windows, hop 1024 (50% overlap), drive-end accelerometer at 12 kHz; the healthy baseline files were decimated from 48 kHz.")
    with tabs[4]:
        if which == 1:
            Q = p1_data("phase1_quick")["summary"]; S = p1["summary"]
            if Q is None:
                st.write("No quick-run files found.")
            else:
                st.caption("Quick run: 1 repeat, class weights instead of SMOTE-NC. It shows how much the resampling choice matters.")
                a = S.pivot(index="classifier", columns="fault", values="mcc").reindex(C.ORDER)[C.FAULTS]
                b = Q.pivot(index="classifier", columns="fault", values="mcc").reindex(C.ORDER)[C.FAULTS]
                c1, c2 = st.columns(2)
                with c1: st.markdown("Full run (SMOTE-NC, MCC)"); st.dataframe(a.round(3))
                with c2: st.markdown("Quick run (class weights, MCC)"); st.dataframe(b.round(3))
                st.markdown("Difference (full − quick)"); show(fig_heat((a - b).round(3), "MCC difference", fmt="{:+.2f}", vmax=0.5))
                qr = p1_data("phase1_quick")["recommendation"]
                if qr is not None: st.markdown("Quick-run recommendation (no MCC floor applied):"); st.dataframe(qr, hide_index=True)
        else:
            a = C.mean_table(p1["pooled"], "mcc", "fault").mean(axis=1).rename("AI4I mean MCC")
            b = C.mean_table(p2, "mcc", "cond").mean(axis=1).rename("CWRU mean MCC")
            t = pd.concat([a, b], axis=1); st.dataframe(t.round(3)); show(fig_heat(t, "Average MCC across all conditions", vmax=1.0))
            st.caption("AI4I faults are generated from known rules of the sensor values, so tree boosters excel on HDF, PWF and OSF. "
                       "CWRU windows carry a physical impact signature, so even logistic regression does well; ranks differ between data sets.")


def page_paradox():
    p1 = p1_data(); S = p1["summary"]
    st.title("Accuracy paradox")
    st.caption("How far does overall accuracy mislead the choice of classifier? (Objective O7.)")
    base = pd.Series({f: 1 - POS[f] / 10000 for f in C.FAULTS})
    S = S.assign(baseline=S.fault.map(base))
    fig, ax = plt.subplots(figsize=(7.2, 4))
    for c in C.ORDER:
        g = S[S.classifier == c]; ax.scatter(g.accuracy, g.mcc, color=CLR[c], label=c, s=40)
    for f in C.FAULTS:
        ax.axvline(base[f], color="#bbb", lw=.8); ax.text(base[f], 1.01, f, fontsize=7, ha="center")
    ax.set_xlabel("Accuracy"); ax.set_ylabel("MCC"); ax.set_ylim(-0.05, 1.05); ax.legend(frameon=False, ncol=6, fontsize=8, loc="lower left")
    ax.set_title("Accuracy cannot tell a good detector from a useless one", loc="left", fontsize=11); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); show(fig)
    st.caption("Grey lines: accuracy of a model that never flags a fault. Every point on or near those lines is almost indistinguishable by accuracy, yet MCC ranges from 0 to 0.94.")
    n_below = int((S.accuracy < S.baseline).sum())
    st.metric("Cells below the always-negative accuracy", f"{n_below} of {len(S)}")
    st.subheader("Ranking distortion per fault")
    rows = []
    for f in C.FAULTS:
        s = S[S.fault == f].set_index("classifier").reindex(C.ORDER)
        for m in ["recall", "f1", "gmean", "mcc"]:
            rows.append(dict(fault=f, metric=C.METRICS[m], **C.distortion(s.accuracy, s[m])))
    D = pd.DataFrame(rows)
    st.dataframe(D.round(3), hide_index=True)
    st.caption("tau: Kendall rank correlation between the accuracy ranking and the metric ranking (1 = identical, negative = reversed). "
               "Displacement: mean rank change (max 3 for six classifiers). DSR: spread of the metric divided by spread of accuracy.")
    st.subheader("Try it yourself")
    c1, c2 = st.columns(2)
    with c1:
        prev = st.slider("Fault prevalence (%)", 0.1, 50.0, 2.0, 0.1); rec = st.slider("Recall (%)", 0, 100, 85)
        spec = st.slider("Specificity (%)", 80.0, 100.0, 98.0, 0.1)
        sc = C.scenario(prev / 100, rec / 100, spec / 100)
    with c2:
        st.dataframe(pd.DataFrame({"metric": ["Accuracy", "Recall", "Precision", "F1", "G-Mean", "MCC"],
                                   "value": [sc["accuracy"], sc["recall"], sc["precision"], sc["f1"], sc["gmean"], max(sc["mcc"], 0)]}).round(3), hide_index=True)
        st.write(f"Of 10,000 samples: {sc['tp']} caught, {sc['fn']} missed, {sc['fp']} false alarms, {sc['tn']} correct.")
    st.write(f"A model that never flags a fault already scores {100 - prev:.1f}% accuracy here, with MCC 0. This model scores "
             f"{100 * sc['accuracy']:.1f}% accuracy and MCC {sc['mcc']:.2f}.")


def page_stats():
    p1, p2 = p1_data(), p2_data()
    st.title("Statistics and recommendation")
    t1, t2, t3, t4 = st.tabs(["Friedman and Nemenyi", "Recommender", "Floor sensitivity", "Tuning (O6)"])
    with t1:
        for label, tab in [("AI4I, blocks = 5 faults", C.mean_table(p1["pooled"], "mcc", "fault")[C.FAULTS]),
                           ("CWRU, blocks = 5 conditions", C.mean_table(p2, "mcc", "cond")[C.cond_order(p2)])]:
            M = tab.to_numpy().T; chi2, p, ranks, cd = C.friedman_nemenyi(M)
            st.markdown(f"**{label}:** χ² = {chi2:.2f}, p = {p:.3f}, critical difference {cd:.2f}")
            show(fig_cd(ranks, list(tab.index), cd, f"Mean ranks by MCC, {label}"))
        st.caption("With only five blocks the test has little power. Classifiers whose mean ranks differ by less than the critical difference cannot be separated. "
                   "That is why the recommendation is made per fault, from fold-level evidence.")
    with t2:
        floor = st.slider("Minimum MCC for any recommendation", 0.0, 0.5, C.MCC_FLOOR, 0.01, key="floor")
        rec = recommendation(floor)
        st.markdown("**Thesis rule on AI4I (fold-level Bayesian test, parsimony order LR, DT, SVM, RF, LGBM, XGB):**")
        st.dataframe(rec, hide_index=True)
        k = st.selectbox("Inspect a fault", C.FAULTS, key="recf"); best, pw = pairwise(k)
        st.dataframe(pw.round(3), hide_index=True)
        st.markdown("**Quick tolerance rule (both data sets)**")
        tol = st.slider("Tolerance below the best mean MCC", 0.0, 0.15, 0.03, 0.005)
        rows = []
        for lab, tab in [("AI4I", C.mean_table(p1["pooled"], "mcc", "fault")[C.FAULTS]), ("CWRU", C.mean_table(p2, "mcc", "cond")[C.cond_order(p2)])]:
            for col in tab.columns:
                pick, ok = C.tolerance_pick(tab[col], tol, floor)
                rows.append(dict(dataset=lab, task=col, best_mcc=round(tab[col].max(), 3), recommended=pick or "none", acceptable=" ".join(ok)))
        st.dataframe(pd.DataFrame(rows), hide_index=True)
    with t3:
        out = pd.DataFrame({f"floor {fl:g}": recommendation(fl).set_index("fault").recommended for fl in (0.05, 0.10, 0.20)})
        st.dataframe(out); st.caption("A recommendation that survives all three floors does not depend on the floor choice.")
    with t4:
        import os
        for name in ("tuned_stability.csv",):
            p = os.path.join(C.DATA, name)
            if os.path.exists(p):
                st.dataframe(pd.read_csv(p), hide_index=True)
            else:
                st.info("No tuned_stability.csv in the data folder yet. Run phase1_tune.py (nested cross-validation, Table 3.6 grids) and copy "
                        "its tuned_stability.csv, tuned_pooled.csv and tuned_params.csv into data/ to show tuned scores and ranking stability here.")
        pp = os.path.join(C.DATA, "tuned_params.csv")
        if os.path.exists(pp): st.markdown("Selected hyperparameters per outer fold"); st.dataframe(pd.read_csv(pp))


def page_features():
    w = windows(); st.title("Signal features")
    st.caption("Ten time-domain features per 2048-sample window. Impulsive features (kurtosis, crest, impulse factor) respond to bearing defects.")
    f = st.selectbox("Feature", C.CWRU_FEATS, index=C.CWRU_FEATS.index("kurt"))
    order = ["Normal", "IR", "B", "OR"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bp = ax.boxplot([w[w.fault_type == t][f] for t in order], patch_artist=True, showfliers=False)
    ax.set_xticklabels(["Healthy", "Inner race", "Ball", "Outer race"])
    for p, c in zip(bp["boxes"], ["#9aa5b1", "#1f77b4", "#ff7f0e", "#2ca02c"]): p.set_facecolor(c); p.set_alpha(.6)
    ax.set_title(f"{f} by bearing condition", loc="left", fontsize=11); ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); show(fig)
    st.dataframe(w.groupby("fault_type")[C.CWRU_FEATS].mean().reindex(order).round(3))


def model_card(b, kind):
    st.markdown("**Model card (out-of-fold, data the model did not train on)**")
    a, o = b["at_half"], b["at_opt"]
    t = pd.DataFrame({"threshold 0.50": a, f"chosen threshold {b['threshold']:.2f}": o}).rename(index=C.METRICS)
    st.dataframe(t.round(3))
    st.write(f"ROC-AUC {b['auc']:.3f} · average precision {b['ap']:.3f} · {b['n_pos']} positives in {b['n']} {kind} · "
             f"{'SMOTE-resampled training folds' if b['smote'] else 'class weights (imbalanced-learn not installed)'}")
    sw = C.threshold_sweep(b["y"], b["oof"]); show(fig_sweep(sw, b["threshold"], "Recall, precision and MCC against the threshold"))
    st.markdown("Permutation importance (drop in average precision when a feature is shuffled)")
    st.bar_chart(b["importance"].round(4))


def page_predict_ai4i():
    st.title("Predict · AI4I machine")
    df, src = ai4i_df()
    if df is None:
        st.warning("The raw AI4I 2020 file is needed to train the live models. " + str(src))
        up = st.file_uploader("Upload ai4i2020.csv (or the UCI zip)", type=["csv", "zip"])
        if up:
            try:
                st.session_state["_ai4i_df"] = C.read_ai4i(up.read()); st.session_state["_ai4i_src"] = "uploaded"; st.rerun()
            except Exception as e:
                st.error(f"Could not read that file: {e}")
        st.stop()
    st.caption(f"Training data: {len(df):,} rows from {src}")
    cap = C.capabilities(); avail = C.available_classifiers()
    S = p1_data()["summary"]; ch = C.champion_table(S).set_index("fault")
    rec = recommendation(C.MCC_FLOOR).set_index("fault")
    with st.expander("Data audit and documented failure rules"):
        st.dataframe(pd.DataFrame({"positives": df[C.FAULTS].sum(), "prevalence %": (df[C.FAULTS].mean() * 100).round(3),
                                   "always-negative accuracy %": ((1 - df[C.FAULTS].mean()) * 100).round(3)}))
        st.markdown("How well the failure conditions documented for this data set reproduce the labels (rule says fault ⇒ flag):")
        st.dataframe(C.rule_audit(df).round(3), hide_index=True)
        st.caption("HDF: temperature gap < 8.6 K and speed < 1380 rpm. PWF: power outside 3,500–9,000 W. OSF: tool wear × torque above 11,000 / 12,000 / 13,000 Nm·min for type L / M / H. "
                   "TWF: tool replaced or failed at random between 200 and 240 min. RNF: random, 0.1% of rows.")
    st.subheader("Enter a machine reading")
    with st.form("ai4i_form"):
        c1, c2, c3 = st.columns(3)
        typ = c1.selectbox("Product quality type", ["L", "M", "H"], index=0)
        air = c1.number_input("Air temperature [K]", 290.0, 310.0, float(round(df.air_temp.median(), 1)), 0.1)
        proc = c2.number_input("Process temperature [K]", 300.0, 320.0, float(round(df.proc_temp.median(), 1)), 0.1)
        rpm = c2.number_input("Rotational speed [rpm]", 1000.0, 3000.0, float(round(df.rot_speed.median())), 1.0)
        tq = c3.number_input("Torque [Nm]", 1.0, 100.0, float(round(df.torque.median(), 1)), 0.1)
        wear = c3.number_input("Tool wear [min]", 0.0, 300.0, float(round(df.tool_wear.median())), 1.0)
        modes = {f: "Champion (best MCC)" for f in C.FAULTS}
        with st.expander("Choose the classifier per fault (default: champion)"):
            for f in C.FAULTS:
                opts = ["Champion (best MCC)", "Thesis recommendation"] + avail   # SVM is slow: it fits probabilities
                modes[f] = st.selectbox(f"{f} · {C.FAULT_NAME[f]}", opts, key=f"mode_{f}")
        go = st.form_submit_button("Assess this reading")
    if not go:
        st.info("Fill in the reading and press the button. The first assessment trains the models (about a minute with SMOTE); later ones are instant.")
    else:
        row = pd.DataFrame([dict(air_temp=air, proc_temp=proc, rot_speed=rpm, torque=tq, tool_wear=wear, type={"L": 0, "M": 1, "H": 2}[typ])])
        st.session_state["ai4i_row"], st.session_state["ai4i_modes"] = row, modes
    if "ai4i_row" in st.session_state:
        row, modes = st.session_state["ai4i_row"], st.session_state.get("ai4i_modes", {})
        derived = pd.DataFrame({"quantity": ["Power [W]", "Temperature gap [K]", "Wear × torque [Nm·min]"],
                                "value": [C.power_w(row.torque, row.rot_speed)[0], (row.proc_temp - row.air_temp)[0], (row.tool_wear * row.torque)[0]]}).round(1)
        st.dataframe(derived, hide_index=True)
        flags = C.rule_flags(row).iloc[0]; results = []
        prog = st.progress(0.0, text="Training and scoring the models")
        for i, f in enumerate(C.FAULTS):
            mode = modes.get(f, "Champion (best MCC)")
            clf = ch.loc[f, "champion"] if mode.startswith("Champion") else (rec.loc[f, "recommended"] if mode.startswith("Thesis") else mode)
            if clf not in avail:
                clf = next((c for c in S[S.fault == f].sort_values("mcc", ascending=False).classifier if c in avail), avail[0])
            if clf not in C.ORDER:                      # thesis rule said "none (...)"
                clf = ch.loc[f, "champion"] if ch.loc[f, "champion"] in avail else avail[0]
            b = ai4i_bundle(f, clf, True, len(df)); pr = C.predict_with(b, row).iloc[0]
            oor = float(C.outside_range(b, row).iloc[0])
            phase1_mcc = float(S[(S.fault == f) & (S.classifier == clf)].mcc.iloc[0])
            results.append(dict(fault=f, name=C.FAULT_NAME[f], classifier=clf, reliability=C.reliability(phase1_mcc), phase1_mcc=round(phase1_mcc, 3),
                                score=round(pr.score, 3), threshold=round(b["threshold"], 2), probability=pr.probability, alarm=int(pr.alarm),
                                rule=bool(flags[f]) if f != "RNF" else None, outside=oor))
            prog.progress((i + 1) / len(C.FAULTS))
        prog.empty(); R = pd.DataFrame(results)
        alarms = R[(R.alarm == 1) & (R.reliability != "Not reliable")]
        if len(alarms):
            st.error("Alarm: " + ", ".join(f"{r.fault} ({r.name})" for r in alarms.itertuples()))
        elif R.rule.fillna(False).any():
            st.warning("The documented rule flags a fault that the models did not alarm: " + ", ".join(R[R.rule.fillna(False)].fault))
        else:
            st.success("No reliable alarm for this reading.")
        if (R.outside > 0).any():
            st.warning("Some inputs lie well outside the range of the training data; treat these predictions as extrapolations.")
        for r in R.itertuples():
            c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1.4])
            c1.markdown(f"**{r.fault}** · {r.name}  \n{r.classifier} · Phase 1 MCC {r.phase1_mcc} ({r.reliability})")
            c2.metric("Fault probability", pct(r.probability)); c3.metric("Alarm", "YES" if r.alarm else "no")
            c4.write(f"Score {r.score} vs threshold {r.threshold}" + (f"  \nDocumented rule: {'FIRES' if r.rule else 'does not fire'}" if r.rule is not None else ""))
            if r.reliability == "Not reliable":
                st.caption(f"{r.fault} is not predictable from these sensor columns (Phase 1 MCC below the 0.10 floor). Ignore this probability.")
            elif r.reliability == "Weak":
                st.caption(f"{r.fault} is only weakly predictable: expect many false alarms per real fault.")
        pick = st.selectbox("Model card for", C.FAULTS, key="mc_ai4i")
        rr = R[R.fault == pick].iloc[0]; model_card(ai4i_bundle(pick, rr.classifier, True, len(df)), "rows")
    st.subheader("Score a whole file")
    up = st.file_uploader("CSV with the AI4I columns (original names or air_temp, proc_temp, rot_speed, torque, tool_wear, type)", type="csv", key="ai4i_batch")
    if up:
        try:
            raw = C._clean_ai4i(pd.read_csv(up)); need = C.AI4I_FEATURES
            miss = [c for c in need if c not in raw]
            if miss: st.error(f"Missing columns: {miss}")
            else:
                if raw["type"].isna().any(): st.error("Type must be L, M or H."); st.stop()
                out = raw.copy(); X = raw[need]
                with st.spinner("Scoring"):
                    for f in C.FAULTS:
                        clf = ch.loc[f, "champion"] if ch.loc[f, "champion"] in avail else avail[0]
                        b = ai4i_bundle(f, clf, True, len(df)); r = C.predict_with(b, X)
                        out[f"{f}_prob"] = r.probability.round(4); out[f"{f}_alarm"] = r.alarm
                st.dataframe(out.head(300)); st.download_button("Download predictions", out.to_csv(index=False), "ai4i_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not score that file: {e}")


def page_predict_cwru():
    st.title("Predict · CWRU bearing")
    w = windows(); p2 = p2_data(); avail = C.available_classifiers()
    st.caption("Three detectors (ball, inner race, outer race), each trained against the healthy windows with the thesis protocol. Give a vibration record and get a verdict per window and per record.")
    c1, c2, c3 = st.columns(3)
    prev_lab = c1.selectbox("Training fault prevalence", ["2%", "5%", "10%", "natural (all windows)"], index=1)
    prev = {"2%": .02, "5%": .05, "10%": .10}.get(prev_lab)
    mode = c2.selectbox("Classifier", ["Champion per fault type"] + avail)
    min_share = c3.slider("Share of windows that must alarm", 0.1, 1.0, 0.5, 0.05)
    clfs = {f: (C.cwru_default_classifier(p2, f, prev) if mode.startswith("Champion") else mode) for f in C.CWRU_FAULT}
    st.write("Classifiers: " + ", ".join(f"{C.CWRU_FAULT[f]} → {c}" for f, c in clfs.items()))
    src = st.radio("Input", ["Upload a vibration record", "Enter feature values", "Use an example window from the data"], horizontal=True)
    F = None; sig = None; fs = C.TARGET_FS
    if src == "Upload a vibration record":
        c1, c2 = st.columns(2)
        fs = c1.selectbox("Sampling rate of the record", [12_000, 24_000, 36_000, 48_000], index=0, format_func=lambda v: f"{v // 1000} kHz")
        up = c2.file_uploader("CSV/TXT (numeric column) or CWRU .mat", type=["csv", "txt", "mat"])
        if up:
            try:
                tab = C.read_signal(up, up.name)
                col = st.selectbox("Signal column", list(tab.columns)) if tab.shape[1] > 1 else tab.columns[0]
                F, sig = C.signal_to_features(tab[col].dropna().to_numpy(), fs)
            except Exception as e:
                st.error(f"Could not use that record: {e}")
    elif src == "Enter feature values":
        med = w[w.fault_type == "Normal"][C.CWRU_FEATS].median()
        with st.form("cwru_feat"):
            cols = st.columns(5); vals = {}
            for i, f in enumerate(C.CWRU_FEATS):
                vals[f] = cols[i % 5].number_input(f, value=float(round(med[f], 4)), format="%.4f", key=f"ft_{f}")
            ok = st.form_submit_button("Assess")
        if ok: st.session_state["cwru_manual"] = pd.DataFrame([vals])
        F = st.session_state.get("cwru_manual")
    else:
        t = st.selectbox("Example from", ["Normal", "IR", "B", "OR"], format_func=lambda v: {"Normal": "Healthy", "IR": "Inner race", "B": "Ball", "OR": "Outer race"}[v])
        n = st.slider("Number of windows", 1, 60, 20)
        F = w[w.fault_type == t].sample(n, random_state=int(st.number_input("Random seed", 0, 999, 0)))[C.CWRU_FEATS].reset_index(drop=True)
    if F is None or len(F) == 0:
        st.info("Provide a record, feature values or an example to assess."); return
    with st.spinner("Training and scoring (cached after the first run)"):
        bundles = {f: cwru_bundle(f, clfs[f], prev, True) for f in C.CWRU_FAULT}
        res = {f: C.predict_with(b, F) for f, b in bundles.items()}
    share = {f: float(r.alarm.mean()) for f, r in res.items()}
    top = max(share, key=share.get)
    oor = float(max(C.outside_range(b, F).mean() for b in bundles.values()))
    if share[top] >= min_share:
        st.error(f"Fault likely: {C.CWRU_FAULT[top]} ({pct(share[top])} of {len(F)} windows alarmed)")
    else:
        st.success(f"No fault pattern: the highest alarm share is {pct(share[top])} ({C.CWRU_FAULT[top]}), below {pct(min_share)}.")
    if oor > 0.2:
        st.warning("These feature values lie far outside the training range (different sensor, scaling or sampling rate?). The detectors were trained on 12 kHz drive-end data; do not trust this verdict.")
    st.dataframe(pd.DataFrame({C.CWRU_FAULT[f]: dict(alarm_share=share[f], median_probability=float(res[f].probability.median()),
                                                     max_probability=float(res[f].probability.max())) for f in res}).T.round(3))
    if len(F) > 1:
        fig, ax = plt.subplots(figsize=(7.2, 3))
        for f, c in zip(res, ["#ff7f0e", "#1f77b4", "#2ca02c"]): ax.plot(res[f].probability.to_numpy(), label=C.CWRU_FAULT[f], color=c, lw=1.5)
        ax.set_ylim(0, 1.02); ax.set_xlabel("Window"); ax.set_ylabel("Fault probability"); ax.legend(frameon=False, ncol=3, loc="upper right")
        ax.spines[["top", "right"]].set_visible(False); fig.tight_layout(); show(fig)
    if sig is not None:
        fig, axs = plt.subplots(1, 2, figsize=(9, 2.8)); t = np.arange(min(len(sig), 4096)) / C.TARGET_FS
        axs[0].plot(t, sig[:len(t)], lw=.6); axs[0].set_xlabel("s"); axs[0].set_title("Signal (first 4096 samples)", fontsize=10)
        X = np.abs(np.fft.rfft(sig[:C.W * 4] - np.mean(sig[:C.W * 4]))); fr = np.fft.rfftfreq(len(sig[:C.W * 4]), 1 / C.TARGET_FS)
        axs[1].plot(fr, X, lw=.6); axs[1].set_xlabel("Hz"); axs[1].set_title("Spectrum", fontsize=10)
        for a in axs: a.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); show(fig)
    with st.expander("Window features"): st.dataframe(F.round(4))
    out = pd.concat([F.reset_index(drop=True)] + [res[f][["probability", "alarm"]].add_prefix(f + "_").reset_index(drop=True) for f in res], axis=1)
    st.download_button("Download window-level results", out.to_csv(index=False), "cwru_windows_scored.csv", "text/csv")
    f = st.selectbox("Model card for", list(C.CWRU_FAULT), format_func=C.CWRU_FAULT.get); model_card(bundles[f], "windows")
    st.caption("Out-of-fold scores come from grouped folds (no vibration record split across training and testing), so they are not inflated by overlapping windows. "
               "Records from other machines, sensors or loads can behave differently from the 12 kHz CWRU records used here.")


# ------------------------------------------------------------------ router
with st.sidebar:
    st.header("Rare Fault Lab")
    page = st.radio("Page", PAGES)
    st.divider()
    df_, src_ = ai4i_df()
    st.caption("AI4I raw data: " + (f"loaded ({len(df_):,} rows)" if df_ is not None else "not found (upload on the AI4I prediction page)"))
    st.caption("Libraries: " + (", ".join(k for k, v in C.capabilities().items() if v) or "scikit-learn only"))

{"Overview": page_overview, "Phase 1 · AI4I results": lambda: page_phase(1), "Phase 2 · CWRU results": lambda: page_phase(2),
 "Accuracy paradox": page_paradox, "Statistics & recommendation": page_stats, "Signal features": page_features,
 "Predict · AI4I machine": page_predict_ai4i, "Predict · CWRU bearing": page_predict_cwru}[page]()
