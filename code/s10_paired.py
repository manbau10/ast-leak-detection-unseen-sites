"""Stage 10. Paired comparisons on the same outer-test files (LOCO by default).

Each model pair: AUC and balanced-accuracy difference with a deployment-cluster paired bootstrap
(2,000 replicates, stratified by label), plus exact McNemar on file correctness (secondary;
files within a deployment are not independent).
Output  tables/paired_<protocol>.csv
Usage   python s10_paired.py REF [OTHER ...] [--protocol=LOCO]
"""
import os
import sys
import pandas as pd

import p5config as C
import metrics as M

COL = os.path.join(C.DERIVED, "pred", "collapsed")


def load(tag, prot, meta):
    c = pd.read_csv(os.path.join(COL, f"{tag}_{prot}.csv"))
    return c.drop(columns=[x for x in ("y", "deployment", "campaign") if x in c]).merge(
        meta[["rid", "y", "deployment"]], on="rid")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    prot = next((a.split("=")[1] for a in sys.argv[1:] if a.startswith("--protocol=")), "LOCO")
    meta = pd.read_csv(os.path.join(C.DERIVED, "retained.csv"))
    ref, others = args[0], args[1:]
    c1 = load(ref, prot, meta)
    rows = []
    for o in others:
        c2 = load(o, prot, meta)
        a = M.paired_bootstrap(c1, c2, "auc")
        b = M.paired_bootstrap(c1, c2, "balanced_accuracy")
        mb, mc, mp = M.mcnemar(c1, c2)
        rows.append(dict(ref=ref, other=o, protocol=prot, d_auc=a[0], d_auc_lo=a[1], d_auc_hi=a[2], p_auc=a[3],
                         d_ba=b[0], d_ba_lo=b[1], d_ba_hi=b[2], p_ba=b[3], mcnemar_ref_only=mb,
                         mcnemar_other_only=mc, mcnemar_p=mp))
        r = rows[-1]
        print(f"{ref} - {o:22s} dAUC {r['d_auc']:+.3f} [{r['d_auc_lo']:+.3f}, {r['d_auc_hi']:+.3f}] p={r['p_auc']:.3f}  "
              f"dBA {r['d_ba']:+.3f} [{r['d_ba_lo']:+.3f}, {r['d_ba_hi']:+.3f}] p={r['p_ba']:.3f}  "
              f"McNemar {mb}/{mc} p={mp:.3f}", flush=True)
    out = os.path.join(C.TABLES, f"paired_{prot}.csv")
    new = pd.DataFrame(rows)
    if os.path.exists(out):
        old = pd.read_csv(out)
        old = old[~(old.ref.eq(ref) & old.other.isin(others))]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(out, index=False)


if __name__ == "__main__":
    main()
