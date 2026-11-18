"""Stage 13b. Where in frequency does the frozen AST + linear probe (input A, pooled) take its evidence.

For each LOCO fold the linear probe is refitted on the outer-training embeddings with the C that fold
selected (deterministic; the refit is checked against the stored outer-test scores). For every
held-out file the probe's logit is back-propagated through the frozen AST to the normalised log-mel
input. Attribution = gradient x (input - floor), where floor is the normalised Kaldi log floor
(the value of an empty bin). Summed over the 998 valid frames, per mel bin.
Regions  in band bins 0 to 67 (upper edge <= 2,048 Hz), straddle 68 to 69, dead 70 to 127.
Output   derived/ast_attrib.npz  rid, signed (n,128), absolute (n,128), logit (n)
         tables/ast_attrib_summary.json
Usage    python s13b_ast_attrib.py [--device=cuda]
"""
import os
import sys
import ast
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import p5config as C
import p5lib as L
import astio as A
from s4_ast_embed import model

DEV = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--device=")), "cuda")
FLOOR = float(np.log(np.finfo(np.float32).eps))
NV = A.n_valid_frames("A")
IN, STR, DEAD = np.arange(0, 68), np.arange(68, 70), np.arange(70, 128)


def main():
    import s4_ast_embed
    s4_ast_embed.DEV = DEV
    r = L.retained()
    y = r.y.values
    X = np.load(os.path.join(C.DERIVED, "emb", "A_frozen.npz"))["pool"].astype(np.float64)
    sel = pd.read_csv(os.path.join(C.DERIVED, "pred", "ast_A_pool_LP_LOCO_sel.csv")).set_index("fold")
    stored = pd.read_csv(os.path.join(C.DERIVED, "pred", "ast_A_pool_LP_LOCO.csv"))
    stored = stored[stored.seed == stored.seed.min()].set_index("rid")["p"]
    net = model()
    fl = (FLOOR - A.AUDIOSET_MEAN) / (2 * A.AUDIOSET_STD)
    out_r, out_s, out_a, out_l = [], [], [], []
    max_dp, t0 = 0.0, time.time()
    for k, tr, te in L.folds("LOCO"):
        c = ast.literal_eval(sel.loc[k, "config"])["C"]
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(sc.transform(X[tr]), y[tr])
        p_ref = lr.predict_proba(sc.transform(X[te]))[:, 1]
        max_dp = max(max_dp, float(np.abs(p_ref - stored.loc[te].values).max()))
        w = torch.tensor(lr.coef_[0] / sc.scale_, dtype=torch.float32, device=DEV)
        b = float(lr.intercept_[0] - (lr.coef_[0] * sc.mean_ / sc.scale_).sum())
        for i in te:
            fb = A.normalise(A.fbank_A(A.load(r.path[i])), A.AUDIOSET_MEAN, A.AUDIOSET_STD)
            x = fb.unsqueeze(0).to(DEV).requires_grad_(True)
            logit = net(x).pooler_output[0] @ w + b
            logit.backward()
            g = (x.grad[0, :NV] * (x[0, :NV] - fl)).detach()
            out_r.append(i); out_s.append(g.sum(0).cpu().numpy()); out_a.append(g.abs().sum(0).cpu().numpy())
            out_l.append(float(logit.detach()))
        print(f"fold {k} done, {len(out_r)} files, {time.time() - t0:.0f}s", flush=True)
    rid = np.array(out_r); S = np.array(out_s); Aa = np.array(out_a); lg = np.array(out_l)
    np.savez(os.path.join(C.DERIVED, "ast_attrib.npz"), rid=rid, signed=S, absolute=Aa, logit=lg)
    # summary
    yy = y[rid]
    share = lambda M, idx: M[:, idx].sum(1) / M.sum(1)
    lo, ce, hi = A.mel_edges("A")
    summ = dict(n_files=int(len(rid)), refit_max_abs_dp=max_dp,
                logit_check_max_abs=float(np.abs(1 / (1 + np.exp(-lg)) - stored.loc[rid].values).max()),
                completeness_corr=float(np.corrcoef(S.sum(1), lg)[0, 1]))
    for name, idx in (("in_band", IN), ("straddle", STR), ("dead", DEAD)):
        sh = share(Aa, idx)
        summ[f"abs_share_{name}"] = dict(all=float(np.median(sh)), leak=float(np.median(sh[yy == 1])),
                                        no_leak=float(np.median(sh[yy == 0])),
                                        q25=float(np.percentile(sh, 25)), q75=float(np.percentile(sh, 75)))
        summ[f"signed_mean_{name}"] = dict(leak=float(S[yy == 1][:, idx].sum(1).mean()),
                                          no_leak=float(S[yy == 0][:, idx].sum(1).mean()))
    top = np.argsort(-Aa.mean(0))[:10]
    summ["top10_bins_abs"] = [dict(bin=int(j), lo_hz=round(float(lo[j])), hi_hz=round(float(hi[j])),
                                   share=float(Aa.mean(0)[j] / Aa.mean(0).sum())) for j in top]
    json.dump(summ, open(os.path.join(C.TABLES, "ast_attrib_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
