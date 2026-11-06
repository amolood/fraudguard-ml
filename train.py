"""FraudGuard — multi-dataset training & evaluation of financial-cybercrime detectors.

Run:  python -m fraudguard.prepare        (once: builds data/processed/*.parquet)
      python train.py                     (all datasets, ~25 min)
      python train.py --datasets ulb --quick

Protocol (identical for the three datasets)
--------------------------------------------
* Temporal split 60 / 20 / 20 on event time (train / validation / test).
* Training negatives capped at 1,000,000 by random sampling (all positives kept),
  the SAME sample for every algorithm. Ranking metrics are unaffected by
  negative undersampling (verified by the imbalance experiment).
* All choices (algorithm, hyper-parameters, imbalance strategy, thresholds)
  are made on validation; test is scored at the end only.
* Primary metric: PR-AUC (average precision) — appropriate at 0.1–0.8 % prevalence.
  Secondary: ROC-AUC, Recall at 1 % FPR, Recall at 0.1 % FPR.
* Robustness: stochastic models re-trained with 3 seeds; bootstrap CIs and a
  paired bootstrap test vs. the runner-up; Friedman test + average ranks over
  (dataset × seed) blocks across all datasets.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import friedmanchisquare
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from fraudguard import __version__
from fraudguard.datasets import DATASETS, label_ar

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
PROCESSED, MODEL_DIR, REPORT_DIR = ROOT / "data" / "processed", ROOT / "models", ROOT / "reports"
SEEDS = [42, 7, 2026]
NEG_CAP = 1_000_000
REVIEW_FPR, BLOCK_FPR = 0.01, 0.001
ALGOS = ["Logistic Regression", "Decision Tree", "Random Forest", "Neural Network (MLP)", "XGBoost", "LightGBM"]
STOCHASTIC = {"Random Forest", "Neural Network (MLP)", "XGBoost", "LightGBM"}

# Feature groups for ablation. "artifact" = simulator artefacts excluded from the main model.
EXCLUDED = {"paysim": ["err_balance_orig", "err_balance_dest"], "ulb": [], "samld": []}
BEHAVIOURAL = {
    "paysim": ["dest_prior_count", "dest_prior_amount", "dest_hours_since_prev"],
    "ulb": [],
    "samld": ["sender_prior_count", "sender_prior_mean", "amount_to_sender_mean", "sender_secs_since_prev",
              "sender_distinct_receivers", "receiver_prior_count", "receiver_distinct_senders",
              "receiver_secs_since_prev", "pair_prior_count"],
}
# Extra ablations: name -> features removed from the main set.
EXTRA_ABLATIONS = {
    "paysim": {"without_balance_fields": ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
                                          "amount_to_balance", "orig_emptied", "dest_zero_both"]},
    "ulb": {"without_amount_time": ["Amount", "log_amount", "hour"]},
    "samld": {},
}
LGB_GRID = [{"num_leaves": 15}, {"num_leaves": 31}, {"num_leaves": 63, "min_child_samples": 300}]
XGB_GRID = [{"max_depth": 4}, {"max_depth": 6}, {"max_depth": 8, "min_child_weight": 10}]


def log(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def preprocessor(numeric: list[str], categorical: list[str], scale: bool) -> ColumnTransformer:
    num = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num.append(("scale", StandardScaler()))
    parts = [("num", Pipeline(num), numeric)]
    if categorical:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=20),
                      categorical))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def make_model(algo: str, numeric, categorical, seed=42, params=None, sampler="none", pos_weight=1.0):
    params = dict(params or {})
    scale = algo in ("Logistic Regression", "Neural Network (MLP)")
    clf = {
        "Logistic Regression": lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000),
        "Decision Tree": lambda: DecisionTreeClassifier(max_depth=10, min_samples_leaf=50, class_weight="balanced",
                                                        random_state=seed),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt",
                                                        max_samples=0.3, class_weight="balanced_subsample",
                                                        n_jobs=-1, random_state=seed),
        "Neural Network (MLP)": lambda: MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=1024,
                                                      learning_rate_init=1e-3, max_iter=15, random_state=seed),
        "XGBoost": lambda: xgb.XGBClassifier(**{**dict(n_estimators=600, learning_rate=0.05, subsample=0.8,
                                             colsample_bytree=0.8, tree_method="hist", eval_metric="aucpr",
                                             n_jobs=-1, random_state=seed, scale_pos_weight=pos_weight), **params}),
        "LightGBM": lambda: lgb.LGBMClassifier(**{**dict(n_estimators=600, learning_rate=0.05, min_child_samples=100,
                                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                               reg_lambda=1.0, n_jobs=-1, random_state=seed, verbose=-1,
                                               scale_pos_weight=pos_weight), **params}),
    }[algo]()
    steps = [("prep", preprocessor(numeric, categorical, scale))]
    if sampler == "undersample":
        steps.append(("sampler", RandomUnderSampler(sampling_strategy=0.1, random_state=seed)))
    elif sampler == "smote":
        steps.append(("sampler", SMOTE(sampling_strategy=0.1, random_state=seed)))
    steps.append(("clf", clf))
    return ImbPipeline(steps)


# ---------------------------------------------------------------------------
# Metrics & statistics
# ---------------------------------------------------------------------------
def recall_at_fpr(y, s, target):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(target, fpr, tpr))


def threshold_at_fpr(y, s, target):
    return float(np.quantile(np.asarray(s)[np.asarray(y) == 0], 1 - target))


def metrics(y, s) -> dict:
    return {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
            "recall_at_1fpr": recall_at_fpr(y, s, 0.01), "recall_at_01fpr": recall_at_fpr(y, s, 0.001)}


def at_threshold(y, s, t) -> dict:
    y, p = np.asarray(y), (np.asarray(s) >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    return {"threshold": float(t), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "recall": tp / max(1, tp + fn), "precision": tp / max(1, tp + fp), "fpr": fp / max(1, fp + tn),
            "alerts_per_10k": 1e4 * (tp + fp) / len(y)}


def r4(d: dict) -> dict:
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in d.items()}


def bootstrap(y, s, s_ref=None, n=300, seed=42) -> dict:
    """Stratified bootstrap (positives and negatives resampled separately)."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    a, d = [], []
    for _ in range(n):
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        a.append(average_precision_score(y[i], s[i]))
        if s_ref is not None:
            d.append(a[-1] - average_precision_score(y[i], np.asarray(s_ref)[i]))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]  # noqa: E731
    out = {"pr_auc_95ci": ci(a), "n_bootstrap": n}
    if d:
        d = np.asarray(d)
        out.update(diff_pr_auc_95ci=ci(d), diff_p_value=round(float(min(1, 2 * min((d <= 0).mean(), (d >= 0).mean()))), 4))
    return out


def psi(e: np.ndarray, a: np.ndarray, bins=10) -> float:
    e, a = e[~np.isnan(e)], a[~np.isnan(a)]
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3 or len(a) == 0:
        return 0.0
    pe = np.clip(np.histogram(np.clip(e, edges[0], edges[-1]), edges)[0] / len(e), 1e-4, None)
    pa = np.clip(np.histogram(np.clip(a, edges[0], edges[-1]), edges)[0] / len(a), 1e-4, None)
    return float(np.sum((pa - pe) * np.log(pa / pe)))


# ---------------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------------
def run_dataset(key: str, quick: bool) -> dict:
    ds = DATASETS[key]
    t0 = time.time()
    df = pd.read_parquet(PROCESSED / f"{key}.parquet")
    if quick:
        df = pd.concat([df[df.y == 1], df[df.y == 0].sample(frac=0.1, random_state=42)]).sort_values("t")
    numeric = [c for c in ds.numeric if c not in EXCLUDED[key]]
    cats = ds.categorical
    feats = numeric + cats
    tr_full, va, te = (df[df.split == s] for s in ("train", "validation", "test"))
    neg = tr_full[tr_full.y == 0]
    tr = pd.concat([tr_full[tr_full.y == 1], neg.sample(min(len(neg), NEG_CAP), random_state=42)]).sort_values("t")
    pos_w = float((tr.y == 0).sum() / max(1, (tr.y == 1).sum()))
    info = {
        "key": key, "title_ar": ds.title_ar, "crime_ar": ds.crime_ar, "source": ds.source, "citation": ds.citation,
        "rows": int(len(df)), "positives": int(df.y.sum()), "prevalence": round(float(df.y.mean()), 5),
        "features_used": feats, "features_excluded": EXCLUDED[key],
        "split": {n: {"rows": int(len(d)), "positives": int(d.y.sum()), "prevalence": round(float(d.y.mean()), 5)}
                  for n, d in (("train", tr_full), ("validation", va), ("test", te))},
        "train_rows_used": int(len(tr)),
    }
    log(f"[{key}] train {len(tr):,} (of {len(tr_full):,}) / val {len(va):,} / test {len(te):,}")

    val_runs, fitted = [], {}

    def fit_eval(name, model, group, data=tr, cols=None):
        cols = cols or feats
        s = time.time()
        model.fit(data[cols], data.y)
        sv = model.predict_proba(va[cols])[:, 1]
        row = r4({"name": name, "group": group, **metrics(va.y, sv), "fit_seconds": time.time() - s})
        val_runs.append(row)
        log(f"[{key}]   {name:<42} val PR-AUC {row['pr_auc']:.4f}  ROC {row['roc_auc']:.4f}  ({row['fit_seconds']:.0f}s)")
        return row

    # 1. Default algorithms
    for a in ALGOS:
        fitted[a] = make_model(a, numeric, cats)
        fit_eval(a, fitted[a], "algorithm")

    # 2. Tune both boosting libraries (fair comparison)
    best_cfg = {}
    for algo, grid in (("LightGBM", LGB_GRID), ("XGBoost", XGB_GRID)):
        cands = []
        for p in (grid[:1] if quick else grid):
            m = make_model(algo, numeric, cats, params=p)
            cands.append((fit_eval(f"{algo} {json.dumps(p)}", m, "tuning")["pr_auc"], p, m))
        score, p, m = max(cands, key=lambda c: c[0])
        best_cfg[algo] = p
        fitted[f"{algo} (tuned)"] = m
    best_algo = max(("LightGBM", "XGBoost"), key=lambda a: max(r["pr_auc"] for r in val_runs if r["name"].startswith(a)))
    log(f"[{key}] best boosting: {best_algo} {best_cfg[best_algo]}")

    # 3. Imbalance strategies on the best boosting model
    imbalance = {"none": max(r["pr_auc"] for r in val_runs if r["name"] == f"{best_algo} {json.dumps(best_cfg[best_algo])}")}
    final_sampler, final_weight = "none", 1.0
    for strat, sampler, w in (("class_weight", "none", pos_w), ("undersample_1:10", "undersample", 1.0),
                              ("smote_1:10", "smote", 1.0)):
        r = fit_eval(f"{best_algo} + {strat}", make_model(best_algo, numeric, cats, params=best_cfg[best_algo],
                                                          sampler=sampler, pos_weight=w), "imbalance")
        imbalance[strat] = r["pr_auc"]
    best_strategy = max(imbalance, key=imbalance.get)
    if best_strategy != "none":
        final_sampler = {"class_weight": "none", "undersample_1:10": "undersample", "smote_1:10": "smote"}[best_strategy]
        final_weight = pos_w if best_strategy == "class_weight" else 1.0

    # 4. Feature ablation
    ablation = {"main": imbalance["none"]}
    if BEHAVIOURAL[key]:
        num_nb = [c for c in numeric if c not in BEHAVIOURAL[key]]
        ablation["without_behavioural"] = fit_eval(
            f"{best_algo} − behavioural features", make_model(best_algo, num_nb, cats, params=best_cfg[best_algo]),
            "ablation", cols=num_nb + cats)["pr_auc"]
    for abl, drop in EXTRA_ABLATIONS[key].items():
        num_x = [c for c in numeric if c not in drop]
        ablation[abl] = fit_eval(f"{best_algo} − {abl}", make_model(best_algo, num_x, cats, params=best_cfg[best_algo]),
                                 "ablation", cols=num_x + cats)["pr_auc"]
    if EXCLUDED[key]:
        num_art = numeric + EXCLUDED[key]
        ablation["with_simulator_artifacts"] = fit_eval(
            f"{best_algo} + simulator artefacts", make_model(best_algo, num_art, cats, params=best_cfg[best_algo]),
            "ablation", cols=num_art + cats)["pr_auc"]

    # 5. Final model (seed 42) + thresholds + calibration from validation
    final_name = f"{best_algo} (tuned{'' if best_strategy == 'none' else ', ' + best_strategy})"
    final = make_model(best_algo, numeric, cats, params=best_cfg[best_algo], sampler=final_sampler, pos_weight=final_weight)
    final.fit(tr[feats], tr.y)
    s_val = final.predict_proba(va[feats])[:, 1]
    thr = {"review": threshold_at_fpr(va.y, s_val, REVIEW_FPR), "block": threshold_at_fpr(va.y, s_val, BLOCK_FPR)}
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s_val, va.y)

    # 6. Test: every algorithm × seeds (tuned configs for the boosting libraries)
    log(f"[{key}] test evaluation with seeds {SEEDS} ...")
    test_runs, test_scores = [], {}
    for a in ALGOS:
        params = best_cfg.get(a, {})
        for seed in (SEEDS if a in STOCHASTIC and not quick else SEEDS[:1]):
            m = make_model(a, numeric, cats, seed=seed, params=params)
            m.fit(tr[feats], tr.y)
            s = m.predict_proba(te[feats])[:, 1]
            test_runs.append(r4({"algorithm": a, "seed": seed, **metrics(te.y, s)}))
            if seed == 42:
                test_scores[a] = s
    runs = pd.DataFrame(test_runs)
    comp = runs.groupby("algorithm").agg(pr_auc=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                                         roc_auc=("roc_auc", "mean"), recall_at_1fpr=("recall_at_1fpr", "mean"),
                                         recall_at_01fpr=("recall_at_01fpr", "mean"), n_seeds=("seed", "size")).fillna(0)
    comparison = {a: r4(v) for a, v in comp.sort_values("pr_auc", ascending=False).to_dict("index").items()}
    s_te = final.predict_proba(te[feats])[:, 1]
    runner_up = next(a for a in comparison if a != best_algo)
    final_test = r4({**metrics(te.y, s_te), "brier_calibrated": float(np.mean((calibrator.predict(s_te) - te.y) ** 2))})
    final_test.update(bootstrap(te.y, s_te, test_scores[runner_up], n=100 if quick else 300))
    ops = {"review_1fpr": r4(at_threshold(te.y, s_te, thr["review"])), "block_01fpr": r4(at_threshold(te.y, s_te, thr["block"]))}

    # 7. Drift: performance over four consecutive test periods + PSI
    periods = []
    for i, chunk in enumerate(np.array_split(np.arange(len(te)), 4), 1):
        d, s = te.iloc[chunk], s_te[chunk]
        row = {"period": i, "rows": int(len(d)), "positives": int(d.y.sum()), "prevalence": float(d.y.mean())}
        if d.y.nunique() == 2:
            row.update(metrics(d.y, s))
            row["fpr_at_review"] = at_threshold(d.y, s, thr["review"])["fpr"]
            row["recall_at_review"] = at_threshold(d.y, s, thr["review"])["recall"]
        periods.append(r4(row))
    drift = sorted(({"feature": c, "label_ar": label_ar(ds, c),
                     "psi": round(psi(tr_full[c].to_numpy(float), te[c].to_numpy(float)), 4)} for c in numeric),
                   key=lambda r: -r["psi"])

    # 8. Explainability (exact TreeSHAP from the booster)
    prep, clf = final.named_steps["prep"], final.named_steps["clf"]
    names = list(prep.get_feature_names_out())
    Z = prep.transform(te[feats].sample(min(20000, len(te)), random_state=42))
    contrib = (clf.booster_.predict(Z, pred_contrib=True) if best_algo == "LightGBM"
               else clf.get_booster().predict(xgb.DMatrix(Z), pred_contribs=True))[:, :-1]
    shap_imp = sorted(({"feature": n, "label_ar": label_ar(ds, n.replace("missingindicator_", "")),
                        "mean_abs_shap": round(float(v), 5)} for n, v in zip(names, np.abs(contrib).mean(0))),
                      key=lambda r: -r["mean_abs_shap"])[:20]

    # 9. Latency
    one = te[feats].iloc[[0]]
    lat = []
    for _ in range(100):
        s = time.perf_counter(); final.predict_proba(one); lat.append((time.perf_counter() - s) * 1000)
    s = time.perf_counter(); final.predict_proba(te[feats].iloc[:20000]); bms = (time.perf_counter() - s) * 1000
    latency = {"single_ms_median": round(float(np.median(lat)), 2), "throughput_per_second": int(20000 / (bms / 1000))}

    # 10. Demo samples + form schema
    demo = te.assign(score=s_te)
    samples = []
    for label, frame in (("احتيال مؤكد — خطورة عالية", demo[demo.y == 1].nlargest(1, "score")),
                         ("احتيال مؤكد — حالة حدّية", demo[(demo.y == 1) & (demo.score >= thr["review"]) & (demo.score < thr["block"])].head(1)),
                         ("احتيال فائت (لم يُكتشف)", demo[(demo.y == 1) & (demo.score < thr["review"])].head(1)),
                         ("عملية سليمة — خطورة منخفضة", demo[demo.y == 0].nsmallest(1, "score")),
                         ("عملية سليمة — إنذار كاذب", demo[(demo.y == 0) & (demo.score >= thr["review"])].head(1))):
        if len(frame):
            r = frame.iloc[0]
            samples.append({"label": label, "actual": int(r.y),
                            "features": {c: (str(r[c]) if c in cats else (None if pd.isna(r[c]) else float(r[c]))) for c in feats}})
    schema = []
    for c in feats:
        f = {"name": c, "label_ar": label_ar(ds, c)}
        if c in cats:
            f.update(type="category", options=sorted(tr_full[c].astype(str).unique().tolist()), default=tr_full[c].mode()[0])
        else:
            col = tr_full[c]
            f.update(type="binary" if set(col.dropna().unique()) <= {0, 1} else "number",
                     default=None if col.isna().all() else float(col.median()),
                     min=float(np.nanmin(col)), max=float(np.nanmax(col)))
        schema.append(f)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": final, "calibrator": calibrator, "thresholds": thr, "dataset": key,
                 "model_name": final_name, "algorithm": best_algo, "numeric": numeric, "categorical": cats,
                 "features": feats, "feature_names_out": names, "version": __version__,
                 "trained_at": datetime.now(timezone.utc).isoformat()}, MODEL_DIR / f"{key}.joblib", compress=3)

    result = {**info, "validation_runs": val_runs, "best_configs": best_cfg, "selected": final_name,
              "imbalance": {k: round(v, 4) for k, v in imbalance.items()}, "ablation": {k: round(v, 4) for k, v in ablation.items()},
              "test_comparison": comparison, "test_runs": test_runs, "final_test": final_test, "runner_up": runner_up,
              "thresholds": thr, "operating_points": ops, "periods": periods, "drift_psi": drift[:10],
              "feature_importance": shap_imp, "latency": latency, "schema": schema, "samples": samples,
              "runtime_seconds": round(time.time() - t0, 1)}
    figures(key, result, te.y.to_numpy(), test_scores, s_te, ops)
    log(f"[{key}] done in {result['runtime_seconds']}s — test PR-AUC {final_test['pr_auc']} {final_test['pr_auc_95ci']}")
    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
PAL = ["#2a5bd7", "#e07b24", "#11a37f", "#7c3aed", "#c2410c", "#64748b"]


def figures(key, r, y, scores, s_final, ops):
    out = REPORT_DIR / key
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    order = list(r["test_comparison"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for i, n in enumerate(order):
        p, rc, _ = precision_recall_curve(y, scores[n])
        a1.plot(rc, p, lw=2 if i == 0 else 1, color=PAL[i % 6], label=f"{n} (AP {r['test_comparison'][n]['pr_auc']:.3f})")
        fpr, tpr, _ = roc_curve(y, scores[n])
        a2.plot(fpr, tpr, lw=2 if i == 0 else 1, color=PAL[i % 6], label=n)
    a1.set_xlabel("Recall"); a1.set_ylabel("Precision"); a1.set_title(f"{key}: Precision–Recall (test)"); a1.legend(fontsize=6, frameon=False)
    a2.set_xscale("log"); a2.set_xlim(1e-4, 1); a2.axvline(0.01, ls="--", color="#999", lw=.8)
    a2.set_xlabel("False-positive rate (log)"); a2.set_ylabel("Recall"); a2.set_title(f"{key}: ROC, log-FPR (test)")
    fig.tight_layout(); fig.savefig(out / "fig_pr_roc.png"); plt.close(fig)

    imp = r["feature_importance"][:15][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([x["feature"] for x in imp], [x["mean_abs_shap"] for x in imp], color="#2a5bd7")
    ax.set_xlabel("mean |SHAP|"); ax.set_title(f"{key}: top features"); fig.tight_layout(); fig.savefig(out / "fig_shap.png"); plt.close(fig)

    pr = [p for p in r["periods"] if "pr_auc" in p]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot([p["period"] for p in pr], [p["pr_auc"] for p in pr], marker="o", color="#2a5bd7", label="PR-AUC")
    ax.plot([p["period"] for p in pr], [p["fpr_at_review"] * 10 for p in pr], marker="s", color="#e07b24", label="FPR at review threshold ×10")
    ax.set_xticks([p["period"] for p in pr]); ax.set_xlabel("Test period (chronological quarters)"); ax.legend(frameon=False, fontsize=7)
    ax.set_title(f"{key}: temporal stability"); fig.tight_layout(); fig.savefig(out / "fig_drift.png"); plt.close(fig)

    o = ops["review_1fpr"]
    cm = np.array([[o["tn"], o["fp"]], [o["fn"], o["tp"]]])
    fig, ax = plt.subplots(figsize=(4, 3.6))
    ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center")
    ax.set_xticks([0, 1], ["Legit", "Crime"]); ax.set_yticks([0, 1], ["Legit", "Crime"]); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{key}: review threshold (1% FPR)"); fig.tight_layout(); fig.savefig(out / "fig_confusion.png"); plt.close(fig)


def cross_dataset(results: dict) -> dict:
    """Friedman test + average ranks over (dataset × seed) blocks."""
    blocks = []
    for key, r in results.items():
        runs = pd.DataFrame(r["test_runs"])
        for seed in SEEDS:
            row = {a: runs[(runs.algorithm == a) & (runs.seed == (seed if a in STOCHASTIC else 42))].pr_auc
                   for a in ALGOS}
            if all(len(v) for v in row.values()):
                blocks.append({a: float(v.iloc[0]) for a, v in row.items()})
    B = pd.DataFrame(blocks)
    out = {"blocks": len(B)}
    if len(B) >= 3:
        stat, p = friedmanchisquare(*[B[a] for a in ALGOS])
        ranks = B.rank(axis=1, ascending=False).mean().sort_values()
        k, n = len(ALGOS), len(B)
        cd = 2.850 * np.sqrt(k * (k + 1) / (6 * n))  # Nemenyi, alpha = 0.05, k = 6 (Demšar, 2006)
        out.update(friedman_chi2=round(float(stat), 3), friedman_p=float(f"{p:.3g}"),
                   average_ranks={a: round(float(v), 2) for a, v in ranks.items()}, nemenyi_cd=round(float(cd), 2))
    return out


def summary_figure(results: dict, cross: dict):
    keys = list(results)
    M = np.array([[results[k]["test_comparison"].get(a, {}).get("pr_auc", np.nan) for k in keys] for a in ALGOS])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for (i, j), v in np.ndenumerate(M):
        ax.text(j, i, f"{v:.3f}", ha="center", va="center", color="white" if v > .6 else "black", fontsize=8)
    ax.set_xticks(range(len(keys)), keys); ax.set_yticks(range(len(ALGOS)), ALGOS)
    ax.set_title("Test PR-AUC (mean over seeds)"); fig.colorbar(im, fraction=.04); fig.tight_layout()
    fig.savefig(REPORT_DIR / "fig_summary_heatmap.png"); plt.close(fig)


# ---------------------------------------------------------------------------
def main(datasets: list[str], quick: bool):
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    REPORT_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / "metrics.json"
    store = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"datasets": {}}
    for k in datasets:
        store["datasets"][k] = run_dataset(k, quick)
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(store, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    store["cross_dataset"] = cross_dataset(store["datasets"])
    store.update(version=__version__, trained_at=datetime.now(timezone.utc).isoformat(), quick_run=quick,
                 protocol={"split": "temporal 60/20/20 on event time", "negatives_cap": NEG_CAP, "seeds": SEEDS,
                           "primary_metric": "PR-AUC", "review_threshold": "1% FPR on validation",
                           "block_threshold": "0.1% FPR on validation", "calibration": "isotonic on validation"},
                 environment={"python": platform.python_version(), "sklearn": sklearn.__version__,
                              "lightgbm": lgb.__version__, "xgboost": xgb.__version__})
    path.write_text(json.dumps(store, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    summary_figure(store["datasets"], store["cross_dataset"])
    log(f"cross-dataset: {store['cross_dataset']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    main(a.datasets, a.quick)
