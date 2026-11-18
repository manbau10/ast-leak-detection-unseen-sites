"""Extra figure data (richer result figures). Writes figures/figdata_extra.json and derived/thumbs.npz.

Blocks
  nights     every development deployment: night scores of the AST probe (LOCO outer scores, mean p per
             night, in date order), label, tier, S, P, threshold. Source: collapsed ast_A_pool_LP_LOCO and
             tables/tiers_deployments.csv. No names are written; deployments get a running number.
  roc        pooled LOCO ROC per deep model, and per held-out site (sites with both classes).
  strips     file scores per site per model with the fold threshold; jitter from a fixed seed.
  tsne       2-D t-SNE (perplexity 30, PCA init, seed 0) of standardised development vectors:
             (i) frozen AST pooled embedding, input A; (ii) log Welch spectrum 0 to 2,048 Hz, no model.
             A picture of neighbourhoods only; distances between clusters carry no meaning.
  confusion  TP, FP, TN, FN at the fixed thresholds (LOCO pooled; locked test).
  thumbs     input A and input B log-mel arrays of the leak and no-leak examples (for small pictures).
Run in deep env .
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import p5config as C
import p5lib as L

T, D = C.TABLES, C.DERIVED
COL = os.path.join(D, "pred", "collapsed")
DL = {"ast_A_pool_LP": "AST", "scratch_CNN": "CNN", "scratch_LSTM": "LSTM"}


def load(tag, meta):
    c = pd.read_csv(os.path.join(COL, f"{tag}_LOCO.csv")).drop(columns=["y", "deployment", "campaign"], errors="ignore")
    return c.merge(meta[["rid", "y", "campaign", "deployment", "night", "date"]], on="rid")


def thin(f, t, n=160):
    if len(f) <= n:
        return f.tolist(), t.tolist()
    i = np.unique(np.r_[np.linspace(0, len(f) - 1, n).astype(int), len(f) - 1])
    return f[i].tolist(), t[i].tolist()


def main():
    meta = pd.read_csv(os.path.join(D, "retained.csv"))
    out = {}
    # nights
    c = load("ast_A_pool_LP", meta)
    td = pd.read_csv(os.path.join(T, "tiers_deployments.csv")).set_index("deployment")
    order = ["Major", "Significant", "Moderate", "Borderline", "Not flagged"]
    td["tier_rank"] = td.tier.map({t: i for i, t in enumerate(order)})
    td = td.sort_values(["tier_rank", "S"], ascending=[True, False])
    rows = []
    for k, (dep, r) in enumerate(td.iterrows(), 1):
        g = c[c.deployment == dep].groupby(["night", "date"]).p.mean().reset_index().sort_values("date")
        rows.append(dict(n=k, site=int(r.campaign), y=int(r.y), tier=r.tier, S=float(r.S), P=float(r.P), thr=float(r.thr),
                         nights=[float(v) for v in g.p]))
    out["nights"] = rows
    # roc, strips
    out["roc"], out["strips"] = {}, {}
    rng = np.random.default_rng(20261106)
    for tag, nm in DL.items():
        c = load(tag, meta)
        f, t, _ = roc_curve(c.y, c.p)
        pooled = thin(f, t)
        sites = {}
        for s, g in c.groupby("campaign"):
            if g.y.nunique() > 1:
                fs, ts, _ = roc_curve(g.y, g.p)
                sites[int(s)] = dict(fpr=thin(fs, ts)[0], tpr=thin(fs, ts)[1], auc=float(roc_auc_score(g.y, g.p)))
        out["roc"][nm] = dict(fpr=pooled[0], tpr=pooled[1], auc=float(roc_auc_score(c.y, c.p)), sites=sites)
        out["strips"][nm] = dict(site=c.campaign.astype(int).tolist(), y=c.y.astype(int).tolist(), p=c.p.round(4).tolist(),
                                 thr=c.thr.round(4).tolist(), jitter=rng.uniform(-0.32, 0.32, len(c)).round(3).tolist())
    # t-SNE
    dev = meta[meta.split == "dev"].rid.values
    X1 = np.load(os.path.join(D, "emb", "A_frozen.npz"))["pool"][dev].astype(np.float64)
    P = np.load(os.path.join(D, "psd.npy"))[dev].astype(np.float64)
    fr = np.load(os.path.join(D, "psd_freqs.npy"))
    X2 = np.log10(P[:, (fr >= 20) & (fr <= C.NYQUIST)] + 1e-20)
    out["tsne"] = dict(y=meta.y.values[dev].tolist(), site=meta.campaign.values[dev].tolist())
    for key, X in (("ast", X1), ("spectrum", X2)):
        Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(StandardScaler().fit_transform(X))
        out["tsne"][key] = Z.round(3).tolist()
    # confusion
    dr = pd.read_csv(os.path.join(T, "dev_results.csv"))
    dr = dr[dr.protocol == "LOCO"].set_index("tag")
    lr = pd.read_csv(os.path.join(T, "locked_results.csv")).set_index("model")
    out["confusion"] = {nm: dict(dev={k: int(dr.loc[tag, k]) for k in ("TP", "FP", "TN", "FN")},
                                 locked={k: int(lr.loc[tag, k]) for k in ("TP", "FP", "TN", "FN")}) for tag, nm in DL.items()}
    json.dump(out, open(os.path.join(C.FIGURES, "figdata_extra.json"), "w"))
    # thumbnails
    import astio as A
    r = L.retained()
    th = {}
    for nm, rid in (("leak", 2146), ("noleak", 898)):
        x = A.load(r.path[rid])
        th[f"A_{nm}"] = A.fbank_A(x)[:A.n_valid_frames("A")].numpy().T
        th[f"B_{nm}"] = A.fbank_B(x)[:A.n_valid_frames("B")].numpy().T
    np.savez(os.path.join(D, "thumbs.npz"), **th)
    print("written", list(out))


if __name__ == "__main__":
    main()
