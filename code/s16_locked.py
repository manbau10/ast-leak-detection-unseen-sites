"""Stage 16. The single locked test on Sites 10 to 12. Models, settings, thresholds and tier cut points
are fixed on Sites 1 to 9 before this script is run.

Each scoring mode runs once. It refuses to start if its output already exists.
Fit = all development files; test = all locked files (guard FINAL).
Modes
  ast      deep env    (environment of s6)   ast_A_pool_LP, ast_B_pool_LP, ast_Afill_pool_LP
  scratch  deep env , GPU                    scratch_CNN, scratch_LSTM
  report   deep env                          metrics, paired tests, per site, deployments, tiers
Output  derived/locked/pred_<tag>.csv (rid, seed, p, thr, fold)
        tables/locked_results.csv, locked_paired.csv, locked_by_site.csv, locked_deployments.csv,
        locked_summary.json
"""
import os
import sys
import json
import numpy as np
import pandas as pd

import p5config as C
import guard

LOCK = os.path.join(C.DERIVED, "locked")
PRED = os.path.join(C.DERIVED, "pred")
AST_TAGS = {"ast_A_pool_LP": "A_pool", "ast_B_pool_LP": "B_pool", "ast_Afill_pool_LP": "Afill_pool"}
MAIN = ["ast_A_pool_LP", "scratch_CNN", "scratch_LSTM", "ast_B_pool_LP", "ast_Afill_pool_LP"]
PRIMARY = "ast_A_pool_LP"
DEV_LOCO_CI = (0.827, 0.976)


def split(meta):
    dev = meta.rid[meta.split == "dev"].values
    lock = meta.rid[meta.split == "locked"].values
    guard.check_split(meta, dev, lock, "FINAL")
    return dev, lock


def save(tag, lock, ps, thr):
    path = os.path.join(LOCK, f"pred_{tag}.csv")
    if os.path.exists(path):
        raise SystemExit(f"{path} exists; the locked test is scored once")
    rows = [pd.DataFrame(dict(rid=lock, seed=s, p=p, thr=thr, fold=0)) for s, p in ps]
    pd.concat(rows).to_csv(path, index=False)
    print(tag, "scored", len(lock), "files", len(ps), "seed(s)", flush=True)


def final_cfg(tag):
    return json.load(open(os.path.join(PRED, f"{tag}_FINAL_sel.json")))


def run_ast(meta, dev, lock):
    from s6_ast_heads import Views, fitter
    y = meta.y.values
    for tag, view in AST_TAGS.items():
        j = final_cfg(tag)
        V = Views(view)
        V.final()                        # B view: dev-only normalisation (B_FINAL); no-op for A views
        fp = fitter("LP", V, y, "cpu")
        save(tag, lock, [(C.SEEDS[0], fp(j["config"], dev, lock, C.SEEDS[0]))], j["threshold"])   # LP is deterministic


def run_scratch(meta, dev, lock):
    import s11_scratch as S
    y = meta.y.values
    for m in ("CNN", "LSTM"):
        j = final_cfg(f"scratch_{m}")
        fp = S.fitter(m, S.Data(m), y)
        save(f"scratch_{m}", lock, [(s, fp({}, dev, lock, s)) for s in C.SEEDS], j["threshold"])


# ------------------------------------------------------------------ report
def youden(y, p):
    from sklearn.metrics import roc_curve
    f, t, th = roc_curve(y, p)
    return float(th[np.argmax(t - f)])


def deployments(c, meta, thr):
    d = c.merge(meta[["rid", "night"]], on="rid")
    n = d.groupby(["deployment", "night"]).agg(p=("p", "mean"), y=("y", "first")).reset_index()
    return n.groupby("deployment").agg(S=("p", "median"), P=("p", lambda v: float((v >= thr).mean())),
                                       nights=("p", "size"), y=("y", "first"))


def tier_cuts(meta):
    """Quartiles of S over leaking dev deployments, from LOCO out-of-fold scores of the primary model (FINAL C)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import p5lib as L
    X = np.load(os.path.join(C.DERIVED, "emb", "A_frozen.npz"))["pool"].astype(np.float64)
    y = meta.y.values
    j = final_cfg(PRIMARY)
    rid, p = [], []
    for k, tr, te in L.folds("LOCO"):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=j["config"]["C"], class_weight="balanced", max_iter=5000).fit(sc.transform(X[tr]), y[tr])
        rid.append(te); p.append(m.predict_proba(sc.transform(X[te]))[:, 1])
    c = pd.DataFrame(dict(rid=np.concatenate(rid), p=np.concatenate(p))).merge(meta[["rid", "y", "deployment"]], on="rid")
    dep = deployments(c, meta, j["threshold"])
    return np.percentile(dep.loc[dep.y == 1, "S"], [25, 50, 75]), j["threshold"]


def report(meta):
    import metrics as M
    from sklearn.metrics import roc_auc_score
    TIERS = ["Not flagged", "Borderline", "Moderate", "Significant", "Major"]
    res, cs, by_site, deps, shift = [], {}, [], [], {}
    for tag in MAIN:
        pred = pd.read_csv(os.path.join(LOCK, f"pred_{tag}.csv"))
        point, c = M.evaluate(pred, meta)
        point["model"] = tag
        cs[tag] = c
        shift[tag] = dict(fixed=float(c.thr.iloc[0]), locked_youden=youden(c.y.values, c.p.values))
        for s, g in c.groupby("campaign"):
            b = M.binary(g.y.values, g.d.values)
            by_site.append(dict(model=tag, site=int(s), n=len(g), n_leak=int(g.y.sum()),
                                auc=roc_auc_score(g.y, g.p) if g.y.nunique() > 1 else np.nan,
                                recall=b["recall"], specificity=b["specificity"], balanced_accuracy=b["balanced_accuracy"]))
        dp = deployments(c, meta, c.thr.iloc[0]).assign(model=tag)
        dp["flag"] = (dp.S >= c.thr.iloc[0]).astype(int)
        deps.append(dp.reset_index())
        res.append(point)
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(C.TABLES, "locked_results.csv"), index=False)
    pd.DataFrame(by_site).to_csv(os.path.join(C.TABLES, "locked_by_site.csv"), index=False)
    D = pd.concat(deps)
    D.to_csv(os.path.join(C.TABLES, "locked_deployments.csv"), index=False)
    paired = []
    for tag in MAIN[1:]:
        for met in ("auc", "balanced_accuracy"):
            d, lo, hi, pv = M.paired_bootstrap(cs[PRIMARY], cs[tag], met)
            paired.append(dict(ref=PRIMARY, other=tag, metric=met, diff=d, lo=lo, hi=hi, p=pv))
    pd.DataFrame(paired).to_csv(os.path.join(C.TABLES, "locked_paired.csv"), index=False)
    # tiers for the primary model
    cuts, thr = tier_cuts(meta)
    pdp = D[D.model == PRIMARY].set_index("deployment")
    tier = [TIERS[0] if s < thr else TIERS[1 + int(np.searchsorted(cuts, s, side="right"))] for s in pdp.S]
    pdp = pdp.assign(tier=tier)
    pdp.reset_index().to_csv(os.path.join(C.TABLES, "locked_tiers.csv"), index=False)
    p0 = R.set_index("model").loc[PRIMARY]
    summ = dict(
        primary=dict(auc=p0.auc, lo=p0.auc_lo, hi=p0.auc_hi, balanced_accuracy=p0.balanced_accuracy,
                     recall=p0.recall, specificity=p0.specificity,
                     transfer_lower_bound_above_half=bool(p0.auc_lo > 0.5),
                     inside_dev_loco_ci=bool(DEV_LOCO_CI[0] <= p0.auc <= DEV_LOCO_CI[1])),
        threshold_shift=shift,
        tier_cuts=dict(threshold=thr, c1=cuts[0], c2=cuts[1], c3=cuts[2]),
        tiers=pd.crosstab(pdp.tier, pdp.y).reindex(TIERS).fillna(0).astype(int).to_dict("index"),
        deployments_primary=pdp[["S", "P", "nights", "y", "flag", "tier"]].round(3).reset_index().to_dict("records"),
    )
    json.dump(summ, open(os.path.join(C.TABLES, "locked_summary.json"), "w"), indent=1, default=float)
    cols = ["model", "auc", "auc_lo", "auc_hi", "balanced_accuracy", "recall", "specificity", "mcc"]
    print(R[cols].round(3).to_string(index=False))
    print(pd.DataFrame(paired).round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in summ.items() if k != "deployments_primary"}, indent=1, default=float))


def main():
    meta = pd.read_csv(os.path.join(C.DERIVED, "retained.csv"))
    assert (meta.rid.values == np.arange(len(meta))).all()
    mode = sys.argv[1]
    if mode == "report":
        report(meta)
        return
    dev, lock = split(meta)
    {"ast": run_ast, "scratch": run_scratch}[mode](meta, dev, lock)


if __name__ == "__main__":
    main()
