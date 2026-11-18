"""Stage 13b check, part 2. Which band carries the leak vs no-leak difference in the probe logit.

From derived/ast_ig.npz (integrated gradients, 100 files). Per file, signed IG summed per region.
Contrast = mean over leak files - mean over no-leak files, per region; the regions add up to the
contrast of logit(input) - logit(empty baseline). 95% CI: deployment-cluster bootstrap, stratified by
label (2,000 replicates), as in metrics.py.
Output  tables/ast_ig_contrast.json
"""
import os
import json
import numpy as np
import pandas as pd

import p5config as C
import metrics as M
IN, STR, DEAD = np.arange(0, 68), np.arange(68, 70), np.arange(70, 128)   # as in s13b_ast_attrib.py


def main():
    z = np.load(os.path.join(C.DERIVED, "ast_ig.npz"))
    meta = pd.read_csv(os.path.join(C.DERIVED, "retained.csv")).set_index("rid")
    d = meta.loc[z["rid"], ["y", "deployment"]].reset_index()
    for name, idx in (("in_band", IN), ("straddle", STR), ("dead", DEAD)):
        d[name] = z["ig"][:, idx].sum(1)
    d["total"] = z["f_x"] - z["f_base"]
    cols = ["in_band", "straddle", "dead", "total"]
    contrast = lambda g: g[g.y == 1][cols].mean() - g[g.y == 0][cols].mean()
    obs = contrast(d)
    dep = d.groupby("deployment")["y"].first()
    pos, neg = dep.index[dep == 1].values, dep.index[dep == 0].values
    rows = {k: v.values for k, v in d.groupby("deployment").groups.items()}
    rng = np.random.default_rng(M.BOOT_SEED + 14)
    bs = []
    for _ in range(M.B_REPS):
        pick = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        g = d.loc[np.concatenate([rows[k] for k in pick])]
        c = contrast(g)
        bs.append(np.r_[c.values, c[["in_band", "straddle", "dead"]].values / c["total"]])
    bs = np.array(bs)
    out = dict(n_files=int(len(d)), n_deployments=int(len(dep)), n_leak_deployments=int(len(pos)))
    for j, k in enumerate(cols):
        lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
        out[f"contrast_{k}"] = dict(est=float(obs[k]), lo=float(lo), hi=float(hi))
    for j, k in enumerate(["in_band", "straddle", "dead"]):
        lo, hi = np.percentile(bs[:, 4 + j], [2.5, 97.5])
        out[f"fraction_of_contrast_{k}"] = dict(est=float(obs[k] / obs["total"]), lo=float(lo), hi=float(hi))
    json.dump(out, open(os.path.join(C.TABLES, "ast_ig_contrast.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
