"""Stage 12. Leak evidence tiers (rule fixed in FINDINGS.md, Stage 12, before this script was run).

Tiers rank leak evidence and received intensity per deployment. They are not leak size.
Input   derived/pred/collapsed/ast_A_pool_LP_LOCO.csv, ast_A_pool_LP_LOCO_sel.csv, emb/A_frozen.npz
Output  tables/tiers_deployments.csv   one row per development deployment
        tables/tiers_summary.json      the five checks
"""
import os
import ast
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf

import p5config as C
import p5lib as L

PRED = os.path.join(C.DERIVED, "pred")
COL = os.path.join(PRED, "collapsed")
TIERS = ["Not flagged", "Borderline", "Moderate", "Significant", "Major"]


def lp(X, y, tr, te, c):
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(C=c, class_weight="balanced", max_iter=5000).fit(sc.transform(X[tr]), y[tr])
    return m.predict_proba(sc.transform(X[te]))[:, 1]


def deploy_scores(meta, rids, p, thr):
    d = meta.loc[rids, ["deployment", "night", "y"]].assign(p=p)
    n = d.groupby(["deployment", "night"]).agg(p=("p", "mean"), y=("y", "first")).reset_index()
    return n.groupby("deployment").agg(S=("p", "median"), P=("p", lambda v: float((v >= thr).mean())),
                                       nights=("p", "size"), y=("y", "first"))


def mahal(X, tr_noleak, tr, te):
    pca = PCA(32, random_state=0).fit(X[tr])
    Z = pca.transform(X)
    lw = LedoitWolf().fit(Z[tr_noleak])
    return np.sqrt(lw.mahalanobis(Z[te]))


def tier(S, thr, c):
    if S < thr:
        return TIERS[0]
    return TIERS[1 + int(np.searchsorted(c, S, side="right"))]


def main():
    meta = pd.read_csv(os.path.join(C.DERIVED, "retained.csv")).set_index("rid", drop=False)
    y = meta["y"].values
    X = np.load(os.path.join(C.DERIVED, "emb", "A_frozen.npz"))["pool"].astype(np.float64)
    out = pd.read_csv(os.path.join(COL, "ast_A_pool_LP_LOCO.csv"))
    sel = pd.read_csv(os.path.join(PRED, "ast_A_pool_LP_LOCO_sel.csv")).set_index("fold")
    rows, fold_info = [], []
    for k, tr, te in L.folds("LOCO"):
        c_k = ast.literal_eval(sel.loc[k, "config"])["C"]
        thr = float(sel.loc[k, "threshold"])
        # training-side cut points from inner leave-one-site-out scores
        oof_r, oof_p = [], []
        for _, a, b in L.inner_folds(k):
            oof_r.append(b)
            oof_p.append(lp(X, y, a, b, c_k))
        tr_dep = deploy_scores(meta, np.concatenate(oof_r), np.concatenate(oof_p), thr)
        cuts = np.percentile(tr_dep.loc[tr_dep.y == 1, "S"], [25, 50, 75])
        # outer-test deployments
        o = out[out.fold == k].set_index("rid").loc[te]
        assert np.allclose(o["thr"], thr)
        te_dep = deploy_scores(meta, te, o["p"].values, thr)
        dist = mahal(X, tr[y[tr] == 0], tr, te)
        dd = meta.loc[te, ["deployment", "night"]].assign(m=dist).groupby(["deployment", "night"])["m"].mean()
        te_dep["anomaly"] = dd.groupby(level=0).median()
        te_dep["tier"] = [tier(s, thr, cuts) for s in te_dep.S]
        te_dep["fold"], te_dep["thr"] = k, thr
        te_dep["c1"], te_dep["c2"], te_dep["c3"] = cuts
        te_dep["campaign"] = meta.groupby("deployment")["campaign"].first().loc[te_dep.index]
        rows.append(te_dep)
        fold_info.append(dict(fold=k, threshold=thr, c1=cuts[0], c2=cuts[1], c3=cuts[2],
                              n_train_leak_deployments=int((tr_dep.y == 1).sum()), borderline_empty=bool(cuts[0] <= thr)))
    dep = pd.concat(rows)
    dep.reset_index().to_csv(os.path.join(C.TABLES, "tiers_deployments.csv"), index=False)

    # checks
    summ = dict(folds=fold_info)
    ct = pd.crosstab(dep.tier, dep.y).reindex(TIERS).fillna(0).astype(int)
    summ["deployments_by_tier"] = {t: dict(no_leak=int(ct.loc[t, 0]), leak=int(ct.loc[t, 1])) for t in TIERS}
    summ["persistence_by_tier"] = dep.groupby("tier")["P"].agg(["median", "min", "max"]).reindex(TIERS).round(3).to_dict("index")
    # error concentration (file level)
    f = out.drop(columns=[c for c in ("y", "deployment", "campaign") if c in out]).merge(
        meta.reset_index(drop=True)[["rid", "y", "deployment"]], on="rid")
    f["err"] = (f.p >= f.thr).astype(int) != f.y
    f["tier"] = f.deployment.map(dep.tier)
    e = f[f.err].tier.value_counts().reindex(TIERS).fillna(0).astype(int)
    summ["file_errors_by_tier"] = e.to_dict()
    summ["file_errors_total"] = int(f.err.sum())
    # cross-model stability
    stab = {}
    for other in ("ast_B_pool_LP",):
        c = pd.read_csv(os.path.join(COL, f"{other}_LOCO.csv")).set_index("rid")
        so = deploy_scores(meta, c.index.values, c["p"].values, 0.5)["S"].loc[dep.index]
        stab[other] = dict(all=float(spearmanr(dep.S, so)[0]),
                           leak_only=float(spearmanr(dep.S[dep.y == 1], so[dep.y == 1])[0]))
    summ["spearman_S_vs_other_models"] = stab
    lk = dep[dep.y == 1]
    rho, pv = spearmanr(lk.S, lk.anomaly)
    summ["spearman_S_vs_anomaly_leak"] = dict(rho=float(rho), p=float(pv), n=int(len(lk)))
    rho2, pv2 = spearmanr(dep.S, dep.anomaly)
    summ["spearman_S_vs_anomaly_all"] = dict(rho=float(rho2), p=float(pv2), n=int(len(dep)))
    json.dump(summ, open(os.path.join(C.TABLES, "tiers_summary.json"), "w"), indent=1, default=float)
    print(json.dumps({k: v for k, v in summ.items() if k != "folds"}, indent=1, default=float))
    print(pd.DataFrame(fold_info).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
