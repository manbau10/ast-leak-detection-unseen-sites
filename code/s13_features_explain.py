"""Stage 13a. Which physical descriptors separate leak from no-leak files (no classifier involved).

Part 1  Effect size per feature on development files (Sites 1 to 9 only; locked sites untouched).
        Cliff's delta = 2 AUC - 1 (leak vs no leak). 95% CI by the stratified deployment-cluster
        bootstrap of metrics.py (2,000 replicates). Site consistency: number of sites with both
        classes whose own delta has the same sign as the pooled delta.
Output  tables/feature_effects.csv, tables/feature_explain_summary.json
"""
import os
import ast
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import p5config as C
import p5lib as L
import metrics as M

warnings.filterwarnings("ignore")
PRED = os.path.join(C.DERIVED, "pred")


def auc_cols(X, y):
    """AUC of every column of X against y, ties averaged."""
    r = rankdata(X, axis=0)
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)


def effects(meta, F):
    dev = meta[meta.split == "dev"]
    X, y = F.loc[dev.index].values, dev.y.values
    delta = 2 * auc_cols(X, y) - 1
    dep = dev.groupby("deployment")["y"].first()
    pos, neg = dep.index[dep == 1].values, dep.index[dep == 0].values
    pos_i = {k: np.flatnonzero(dev.deployment.values == k) for k in dep.index}
    rng = np.random.default_rng(M.BOOT_SEED + 13)
    bs = np.empty((M.B_REPS, X.shape[1]))
    for b in range(M.B_REPS):
        pick = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        ix = np.concatenate([pos_i[k] for k in pick])
        bs[b] = 2 * auc_cols(X[ix], y[ix]) - 1
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    same, n_sites = np.zeros(X.shape[1], int), 0
    for s, g in dev.groupby("campaign"):
        if g.y.nunique() < 2:
            continue
        n_sites += 1
        ds = 2 * auc_cols(F.loc[g.index].values, g.y.values) - 1
        same += (np.sign(ds) == np.sign(delta))
    out = pd.DataFrame(dict(feature=F.columns, cliffs_delta=delta, lo=lo, hi=hi,
                            ci_excludes_zero=(lo > 0) | (hi < 0), sites_same_sign=same, sites_both_classes=n_sites))
    out["abs_delta"] = out.cliffs_delta.abs()
    return out.sort_values("abs_delta", ascending=False)


def main():
    meta = pd.read_csv(os.path.join(C.DERIVED, "retained.csv")).set_index("rid", drop=False)
    F = pd.read_csv(os.path.join(C.DERIVED, "features.csv")).set_index("rid").loc[meta.index]
    groups = pd.read_csv(os.path.join(C.DERIVED, "s3_feature_list.csv")).set_index("name")["group"]
    eff = effects(meta, F)
    eff["group"] = eff.feature.map(groups)
    eff.to_csv(os.path.join(C.TABLES, "feature_effects.csv"), index=False)
    summ = dict(
        n_dev_files=int((meta.split == "dev").sum()),
        n_features=int(F.shape[1]),
        n_ci_excludes_zero=int(eff.ci_excludes_zero.sum()),
        top10_effects=eff.head(10)[["feature", "cliffs_delta", "lo", "hi", "sites_same_sign", "sites_both_classes"]]
        .round(3).to_dict("records"),
    )
    json.dump(summ, open(os.path.join(C.TABLES, "feature_explain_summary.json"), "w"), indent=1, default=float)
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
