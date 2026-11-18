"""Metrics and cluster bootstrap for this study.

Scores are averaged over seeds per file before thresholding (the threshold is fixed per fold by
selection). Point estimates are pooled over all outer-test files of a protocol.
Intervals: percentile bootstrap over deployments (the independent unit), resampled within
label strata (leaking and non-leaking deployments separately), 2,000 replicates.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

B_REPS = 2000
BOOT_SEED = 20261104


def binary(y, d):
    tp = int(((d == 1) & (y == 1)).sum()); fp = int(((d == 1) & (y == 0)).sum())
    tn = int(((d == 0) & (y == 0)).sum()); fn = int(((d == 0) & (y == 1)).sum())
    n = tp + fp + tn + fn
    acc = (tp + tn) / n
    rec = tp / (tp + fn) if tp + fn else np.nan
    spe = tn / (tn + fp) if tn + fp else np.nan
    pre = tp / (tp + fp) if tp + fp else np.nan
    f1 = 2 * pre * rec / (pre + rec) if (tp + fp) and (tp + fn) and (pre + rec) else np.nan
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else np.nan
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    kap = (acc - pe) / (1 - pe) if pe < 1 else np.nan
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, accuracy=acc, balanced_accuracy=np.nanmean([rec, spe]),
                precision=pre, recall=rec, specificity=spe, f1=f1, mcc=mcc, kappa=kap)


def all_metrics(y, p, d):
    out = binary(y, d)
    two = len(np.unique(y)) > 1
    out["auc"] = roc_auc_score(y, p) if two else np.nan
    out["pr_auc"] = average_precision_score(y, p) if two else np.nan
    out["brier"] = float(np.mean((p - y) ** 2))
    return out


def collapse(pred):
    """Average scores over seeds. Returns one row per rid with p, thr, fold, sd over seeds."""
    g = pred.groupby("rid")
    out = g.agg(p=("p", "mean"), p_sd=("p", "std"), thr=("thr", "first"), fold=("fold", "first"),
                n_seeds=("seed", "nunique")).reset_index()
    return out


def evaluate(pred, meta, reps=B_REPS, key_metrics=("auc", "pr_auc", "balanced_accuracy", "recall", "specificity",
                                                    "precision", "f1", "mcc")):
    c = collapse(pred).merge(meta[["rid", "y", "deployment", "campaign"]], on="rid")
    y, p = c["y"].values, c["p"].values
    d = (p >= c["thr"].values).astype(int)
    point = all_metrics(y, p, d)
    # per-seed spread of the threshold-free metric
    per_seed = [roc_auc_score(*_yp(pred[pred.seed == s], meta)) for s in sorted(pred.seed.unique())] \
        if len(np.unique(y)) > 1 else []
    point["auc_seed_sd"] = float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0
    point["n_files"] = int(len(c)); point["n_deployments"] = int(c.deployment.nunique())
    # cluster bootstrap
    dep = c.groupby("deployment")["y"].first()
    pos, neg = dep.index[dep == 1].values, dep.index[dep == 0].values
    rows_by_dep = {k: v.values for k, v in c.groupby("deployment").groups.items()}
    rng = np.random.default_rng(BOOT_SEED)
    boot = {m: [] for m in key_metrics}
    for _ in range(reps):
        pick = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        idx = np.concatenate([rows_by_dep[k] for k in pick])
        m = all_metrics(y[idx], p[idx], d[idx])
        for k in key_metrics:
            boot[k].append(m[k])
    for k in key_metrics:
        a = np.array(boot[k], dtype=float)
        point[f"{k}_lo"], point[f"{k}_hi"] = np.nanpercentile(a, [2.5, 97.5])
    return point, c.assign(d=d)


def _yp(pred_one_seed, meta):
    j = pred_one_seed.merge(meta[["rid", "y"]], on="rid")
    return j["y"].values, j["p"].values


def paired_bootstrap(c1, c2, metric="auc", reps=B_REPS):
    """Difference model1 - model2 on the same files, deployment-cluster bootstrap. Returns diff, lo, hi, p (two-sided)."""
    j = c1[["rid", "y", "p", "d", "deployment"]].merge(c2[["rid", "p", "d"]], on="rid", suffixes=("1", "2"))
    y = j["y"].values

    def stat(ix):
        if metric == "auc":
            return roc_auc_score(y[ix], j.p1.values[ix]) - roc_auc_score(y[ix], j.p2.values[ix])
        m1 = binary(y[ix], j.d1.values[ix])[metric]; m2 = binary(y[ix], j.d2.values[ix])[metric]
        return m1 - m2

    obs = stat(np.arange(len(j)))
    dep = j.groupby("deployment")["y"].first()
    pos, neg = dep.index[dep == 1].values, dep.index[dep == 0].values
    rows = {k: v.values for k, v in j.groupby("deployment").groups.items()}
    rng = np.random.default_rng(BOOT_SEED + 1)
    bs = []
    for _ in range(reps):
        pick = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        bs.append(stat(np.concatenate([rows[k] for k in pick])))
    bs = np.array(bs)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    pval = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
    return float(obs), float(lo), float(hi), float(min(pval, 1.0))


def mcnemar(c1, c2):
    """Exact McNemar on file-level correctness. Reported as a secondary test; files are not independent."""
    from scipy.stats import binomtest
    j = c1[["rid", "y", "d"]].merge(c2[["rid", "d"]], on="rid", suffixes=("1", "2"))
    r1, r2 = j.d1 == j.y, j.d2 == j.y
    b, c = int((r1 & ~r2).sum()), int((~r1 & r2).sum())
    return b, c, (binomtest(b, b + c, 0.5).pvalue if b + c else 1.0)
