"""Stage 9. Score every development prediction file with the same metrics and cluster bootstrap.

Input   derived/pred/<tag>_<protocol>.csv
Output  tables/dev_results.csv   one row per (tag, protocol), point estimates and 95% CIs
        derived/pred/collapsed/<tag>_<protocol>.csv  seed-averaged scores and decisions per file
Usage   python s9_summary.py [pattern]
"""
import os
import re
import sys
import glob
import numpy as np
import pandas as pd

import p5config as C
import metrics as M

PRED = os.path.join(C.DERIVED, "pred")
COL = os.path.join(PRED, "collapsed")
os.makedirs(COL, exist_ok=True)


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else "*"
    meta = pd.read_csv(os.path.join(C.DERIVED, "retained.csv"))
    out_p = os.path.join(C.TABLES, "dev_results.csv")
    old = pd.read_csv(out_p) if os.path.exists(out_p) else pd.DataFrame()
    rows = []
    for f in sorted(glob.glob(os.path.join(PRED, f"{pat}.csv"))):
        name = os.path.basename(f)[:-4]
        m = re.match(r"(.+)_(LOCO|LORO|LDGO|RAND)$", name)
        if not m:
            continue
        tag, prot = m.groups()
        pred = pd.read_csv(f)
        point, c = M.evaluate(pred, meta)
        c.to_csv(os.path.join(COL, f"{name}.csv"), index=False)
        point.update(tag=tag, protocol=prot, seeds=int(pred.seed.nunique()))
        rows.append(point)
        print(f"{tag:28s} {prot}  AUC {point['auc']:.3f} [{point['auc_lo']:.3f}, {point['auc_hi']:.3f}]  "
              f"BA {point['balanced_accuracy']:.3f}  rec {point['recall']:.3f}  spec {point['specificity']:.3f}  "
              f"F1 {point['f1']:.3f}  MCC {point['mcc']:.3f}", flush=True)
    new = pd.DataFrame(rows)
    if len(old):
        old = old[~old.set_index(["tag", "protocol"]).index.isin(new.set_index(["tag", "protocol"]).index)]
        new = pd.concat([old, new], ignore_index=True)
    lead = ["tag", "protocol", "seeds", "n_files", "n_deployments"]
    new = new[lead + [c for c in new.columns if c not in lead]].sort_values(["protocol", "tag"])
    new.to_csv(out_p, index=False)


if __name__ == "__main__":
    main()
