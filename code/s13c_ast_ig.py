"""Stage 13b check. Integrated gradients on a fixed sample, to test the gradient x input band shares.

Gradient x input does not sum to the logit for a deep network (s13b: correlation 0.01), so its band
shares show local sensitivity only. Integrated gradients (Sundararajan et al., 2017) do sum to
logit(input) - logit(baseline). Baseline = the same spectrogram with every valid frame set to the
normalised Kaldi floor (an empty recording); padded frames unchanged. 32 midpoint steps.
Sample   50 leak and 50 no-leak development files, drawn once with seed 20261106. Each is explained by
         the probe of the LOCO fold that held its site out (same refit as s13b).
Output   derived/ast_ig.npz, tables/ast_ig_summary.json
"""
import os
import ast
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import p5config as C
import p5lib as L
import astio as A
import s4_ast_embed
from s13b_ast_attrib import FLOOR, NV, IN, STR, DEAD

DEV = "cuda"
STEPS = 32
SEED = 20261106


def main():
    r = L.retained()
    y = r.y.values
    X = np.load(os.path.join(C.DERIVED, "emb", "A_frozen.npz"))["pool"].astype(np.float64)
    sel = pd.read_csv(os.path.join(C.DERIVED, "pred", "ast_A_pool_LP_LOCO_sel.csv")).set_index("fold")
    gx = np.load(os.path.join(C.DERIVED, "ast_attrib.npz"))
    gx_abs = pd.DataFrame(gx["absolute"], index=gx["rid"])
    rng = np.random.default_rng(SEED)
    dev = r[r.split == "dev"]
    pick = np.concatenate([rng.choice(dev.rid[dev.y == c].values, 50, replace=False) for c in (1, 0)])
    net = s4_ast_embed.model()
    fl = (FLOOR - A.AUDIOSET_MEAN) / (2 * A.AUDIOSET_STD)
    alphas = (np.arange(STEPS) + 0.5) / STEPS
    out = dict(rid=[], ig=[], f_x=[], f_base=[])
    t0 = time.time()
    for k, tr, te in L.folds("LOCO"):
        mine = np.intersect1d(pick, te)
        if len(mine) == 0:
            continue
        c = ast.literal_eval(sel.loc[k, "config"])["C"]
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(sc.transform(X[tr]), y[tr])
        w = torch.tensor(lr.coef_[0] / sc.scale_, dtype=torch.float32, device=DEV)
        b = float(lr.intercept_[0] - (lr.coef_[0] * sc.mean_ / sc.scale_).sum())
        f = lambda z: net(z).pooler_output[0] @ w + b
        for i in mine:
            x = A.normalise(A.fbank_A(A.load(r.path[i])), A.AUDIOSET_MEAN, A.AUDIOSET_STD).unsqueeze(0).to(DEV)
            xb = x.clone(); xb[0, :NV] = fl
            gsum = torch.zeros_like(x)
            for a in alphas:
                z = (xb + float(a) * (x - xb)).requires_grad_(True)
                f(z).backward()
                gsum += z.grad
            ig = ((x - xb) * gsum / STEPS)[0, :NV].sum(0)
            with torch.no_grad():
                out["f_x"].append(float(f(x))); out["f_base"].append(float(f(xb)))
            out["rid"].append(int(i)); out["ig"].append(ig.cpu().numpy())
        print(f"fold {k}: {len(out['rid'])} files, {time.time() - t0:.0f}s", flush=True)
    rid = np.array(out["rid"]); IG = np.array(out["ig"]); fx = np.array(out["f_x"]); fb = np.array(out["f_base"])
    np.savez(os.path.join(C.DERIVED, "ast_ig.npz"), rid=rid, ig=IG, f_x=fx, f_base=fb)
    yy = y[rid]
    gap = IG.sum(1) - (fx - fb)
    summ = dict(n_files=int(len(rid)), n_leak=int(yy.sum()), steps=STEPS,
                completeness_abs_err_median=float(np.median(np.abs(gap))),
                completeness_rel_err_median=float(np.median(np.abs(gap) / np.abs(fx - fb))),
                completeness_corr=float(np.corrcoef(IG.sum(1), fx - fb)[0, 1]))
    A_ig = np.abs(IG)
    G = gx_abs.loc[rid].values
    for name, idx in (("in_band", IN), ("straddle", STR), ("dead", DEAD)):
        s_ig = A_ig[:, idx].sum(1) / A_ig.sum(1)
        s_gx = G[:, idx].sum(1) / G.sum(1)
        summ[f"ig_abs_share_{name}"] = dict(all=float(np.median(s_ig)), leak=float(np.median(s_ig[yy == 1])),
                                           no_leak=float(np.median(s_ig[yy == 0])))
        summ[f"gxi_abs_share_{name}_same_files"] = dict(all=float(np.median(s_gx)), leak=float(np.median(s_gx[yy == 1])),
                                                       no_leak=float(np.median(s_gx[yy == 0])))
        summ[f"spearman_ig_vs_gxi_share_{name}"] = float(spearmanr(s_ig, s_gx)[0])
        summ[f"ig_signed_{name}"] = dict(leak=float(IG[yy == 1][:, idx].sum(1).mean()),
                                        no_leak=float(IG[yy == 0][:, idx].sum(1).mean()))
    summ["ig_signed_total_minus_baseline"] = dict(leak=float((fx - fb)[yy == 1].mean()), no_leak=float((fx - fb)[yy == 0].mean()))
    json.dump(summ, open(os.path.join(C.TABLES, "ast_ig_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
