"""Shared helpers for this study."""
import os
import numpy as np
import pandas as pd

import p5config as C


def retained():
    """Retained files in rid order, with private folder and file names attached for reading audio."""
    r = pd.read_csv(os.path.join(C.DERIVED, "retained.csv"))
    pm = pd.read_csv(os.path.join(C.DERIVED, "manifest_private.csv"))
    pm = pm.loc[pm["retained"], ["md5", "folder", "file"]]
    assert pm["md5"].is_unique
    r = r.merge(pm, on="md5", how="left").sort_values("rid").reset_index(drop=True)
    assert r["file"].notna().all() and (r["rid"].values == np.arange(len(r))).all()
    r["path"] = [os.path.join(C.DATA, a, b) for a, b in zip(r["folder"], r["file"])]
    return r


def folds(protocol):
    f = pd.read_csv(os.path.join(C.DERIVED, "folds.csv"))
    f = f[f["protocol"] == protocol]
    out = []
    for k, g in f.groupby("fold"):
        out.append((int(k), np.sort(g.loc[g.role == "train", "rid"].values), np.sort(g.loc[g.role == "test", "rid"].values)))
    return out


def inner_folds(outer_fold):
    f = pd.read_csv(os.path.join(C.DERIVED, "inner.csv"))
    f = f[(f["protocol"] == "LOCO") & (f["outer_fold"] == outer_fold)]
    return [(int(k), np.sort(g.loc[g.role == "train", "rid"].values), np.sort(g.loc[g.role == "test", "rid"].values))
            for k, g in f.groupby("inner_fold")]


# AST patch geometry (verified in s2_ast_check)
F_ROWS, T_COLS = 12, 101
A_INBAND_ROWS = list(range(0, 6))      # rows whose upper mel edge is at or below 2,048 Hz
A_STRADDLE_ROW = 6                     # 1,594 to 2,486 Hz
A_DEAD_ROWS = list(range(7, 12))       # entirely above 2,048 Hz
A_VALID_COLS = list(range(0, 99))      # columns free of zero padding (998 valid frames)
B_VALID_COLS = list(range(0, 100))     # 1,014 valid frames


def patch_index(rows, cols):
    """Token positions (after CLS and distillation) for the given patch rows and columns."""
    return np.array([2 + r * T_COLS + c for r in rows for c in cols])
