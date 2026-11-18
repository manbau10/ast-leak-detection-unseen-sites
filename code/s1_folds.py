"""Stage 1. Fix every fold before any feature or model exists.

Input   derived/manifest.csv (retained rows only)
Output  derived/folds.csv     rid, protocol, fold, role (train / test)
        derived/inner.csv     rid, protocol, outer_fold, inner_fold, role
        derived/s1_report.json

Protocols on the development campaigns (1 to 9). The locked campaigns (10 to 12) are never
in any fold. They are scored once at the end.
  LOCO   leave one campaign out (9 outer folds). Primary.
  LORO   leave one region out (6 outer folds). Stricter. Campaigns in one region share a district.
  LDGO   deployment-grouped, stratified 5-fold (a deployment is never split). Intermediate.
  RAND   file-level stratified 5-fold. Shown only to measure inflation. Never used for selection.
Inner folds (model selection) exist for LOCO only. They are leave-one-campaign-out inside the
eight training campaigns, so selection never sees the outer test campaign.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import p5config as C
from guard import check_split

SEED = 2026


def main():
    m = pd.read_csv(os.path.join(C.DERIVED, "manifest.csv"))
    m = m[m["retained"]].reset_index(drop=True)
    m.insert(0, "rid", np.arange(len(m)))
    m.to_csv(os.path.join(C.DERIVED, "retained.csv"), index=False)
    dev = m[m["split"] == "dev"]
    assert set(dev["campaign"]) == set(C.DEV_CAMPAIGNS)

    rows, inner = [], []

    def add(protocol, fold, tr, te):
        check_split(m, tr, te, protocol)
        rows.extend((r, protocol, fold, "train") for r in tr)
        rows.extend((r, protocol, fold, "test") for r in te)

    # LOCO
    for k, c in enumerate(sorted(dev["campaign"].unique())):
        te = dev.loc[dev["campaign"] == c, "rid"].values
        tr = dev.loc[dev["campaign"] != c, "rid"].values
        add("LOCO", k, tr, te)
        trd = dev[dev["campaign"] != c]
        for j, ci in enumerate(sorted(trd["campaign"].unique())):
            ite = trd.loc[trd["campaign"] == ci, "rid"].values
            itr = trd.loc[trd["campaign"] != ci, "rid"].values
            check_split(m, itr, ite, "LOCO")
            inner.extend((r, "LOCO", k, j, "train") for r in itr)
            inner.extend((r, "LOCO", k, j, "test") for r in ite)
    # LORO
    for k, rg in enumerate(sorted(dev["region"].unique())):
        te = dev.loc[dev["region"] == rg, "rid"].values
        tr = dev.loc[dev["region"] != rg, "rid"].values
        add("LORO", k, tr, te)
    # LDGO
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for k, (a, b) in enumerate(sgk.split(dev, dev["y"], dev["deployment"])):
        add("LDGO", k, dev["rid"].values[a], dev["rid"].values[b])
    # RAND
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for k, (a, b) in enumerate(skf.split(dev, dev["y"])):
        add("RAND", k, dev["rid"].values[a], dev["rid"].values[b])

    f = pd.DataFrame(rows, columns=["rid", "protocol", "fold", "role"])
    f.to_csv(os.path.join(C.DERIVED, "folds.csv"), index=False)
    pd.DataFrame(inner, columns=["rid", "protocol", "outer_fold", "inner_fold", "role"]).to_csv(
        os.path.join(C.DERIVED, "inner.csv"), index=False)

    rep = {}
    for p, g in f[f.role == "test"].groupby("protocol"):
        t = g.merge(m[["rid", "y", "campaign", "region", "deployment"]], on="rid")
        rep[p] = dict(folds=int(t["fold"].nunique()), test_files=int(len(t)),
                      every_dev_file_tested_once=bool(t["rid"].is_unique and len(t) == len(dev)),
                      per_fold=[dict(fold=int(k), files=int(len(x)), leak=int(x["y"].sum()),
                                     deployments=int(x["deployment"].nunique()),
                                     leak_deployments=int(x.groupby("deployment")["y"].first().sum()),
                                     campaigns=sorted(int(v) for v in x["campaign"].unique()))
                                for k, x in t.groupby("fold")])
    rep["locked"] = dict(files=int((m.split == "locked").sum()), leak=int(m.loc[m.split == "locked", "y"].sum()),
                         deployments=int(m.loc[m.split == "locked", "deployment"].nunique()))
    rep["locked_in_any_fold"] = bool(f["rid"].isin(m.loc[m.split == "locked", "rid"]).any())
    assert not rep["locked_in_any_fold"]
    with open(os.path.join(C.DERIVED, "s1_report.json"), "w") as fh:
        json.dump(rep, fh, indent=1)
    for p in ("LOCO", "LORO", "LDGO", "RAND"):
        print(p, rep[p]["folds"], "folds, tested once:", rep[p]["every_dev_file_tested_once"])
        for r in rep[p]["per_fold"]:
            print("   ", r)
    print("locked", rep["locked"], "in any fold:", rep["locked_in_any_fold"])


if __name__ == "__main__":
    main()
