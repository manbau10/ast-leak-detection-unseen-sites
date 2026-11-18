"""Leakage guard. Every train/test split in this study passes through check_split.

Checks
  always      no file on both sides; no locked-campaign file on either side of a dev split
  LOCO        no campaign on both sides (implies no deployment, location or night shared)
  LORO        no region on both sides
  LDGO        no deployment on both sides
  RAND        none beyond the first two (it is the deliberately leaky reference)
  FINAL       train = all dev, test = all locked
Any failure raises, so a leaky result cannot be produced silently.
"""
import numpy as np


class LeakageError(AssertionError):
    pass


def _side(m, idx, col):
    return set(m.loc[m["rid"].isin(idx), col].unique())


def check_split(m, train_idx, test_idx, protocol):
    tr, te = set(map(int, train_idx)), set(map(int, test_idx))
    if not tr or not te:
        raise LeakageError(f"{protocol}: empty side")
    if tr & te:
        raise LeakageError(f"{protocol}: {len(tr & te)} files on both sides")
    locked = set(m.loc[m["split"] == "locked", "rid"])
    if protocol == "FINAL":
        if te != locked or tr & locked:
            raise LeakageError("FINAL: test must be exactly the locked campaigns")
        return True
    if (tr | te) & locked:
        raise LeakageError(f"{protocol}: locked campaign file used in development")
    col = {"LOCO": "campaign", "LORO": "region", "LDGO": "deployment"}.get(protocol)
    if col:
        shared = _side(m, tr, col) & _side(m, te, col)
        if shared:
            raise LeakageError(f"{protocol}: {col} on both sides {sorted(shared)[:5]}")
    # identical audio may never cross a split (duplicates were removed in s0; this re-checks)
    if "md5" in m.columns:
        if _side(m, tr, "md5") & _side(m, te, "md5"):
            raise LeakageError(f"{protocol}: identical audio on both sides")
    return True
