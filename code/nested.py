"""Nested evaluation engine shared by every model family in this study.

For one protocol and one model family:
  for each outer fold
      inner splits are built from the outer TRAINING side only
          LOCO   the precomputed inner leave-one-campaign-out folds (inner.csv)
          LORO   leave-one-region-out inside the outer training regions
          LDGO   4-fold StratifiedGroupKFold by deployment
          RAND   4-fold StratifiedKFold by file
      each candidate configuration is fitted on every inner train and scored on inner test
      the configuration with the best pooled inner ROC AUC is chosen
      the decision threshold is the Youden point of the pooled inner out-of-fold scores
      the chosen configuration is refitted on the full outer training side (one fit per seed)
      the outer test side is scored once
Nothing from an outer test fold reaches selection, thresholding or fitting.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

import p5lib as L
from guard import check_split

INNER_SEED = 7


def inner_splits(protocol, outer_fold, tr, meta):
    if protocol == "LOCO":
        return [(a, b) for _, a, b in L.inner_folds(outer_fold)]
    sub = meta.loc[tr]
    if protocol == "LORO":
        regs = sorted(sub["region"].unique())
        return [(sub.index[sub.region != g].values, sub.index[sub.region == g].values) for g in regs]
    if protocol == "LDGO":
        s = StratifiedGroupKFold(4, shuffle=True, random_state=INNER_SEED)
        return [(sub.index[a].values, sub.index[b].values) for a, b in s.split(sub, sub.y, sub.deployment)]
    if protocol == "RAND":
        s = StratifiedKFold(4, shuffle=True, random_state=INNER_SEED)
        return [(sub.index[a].values, sub.index[b].values) for a, b in s.split(sub, sub.y)]
    raise ValueError(protocol)


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    k = int(np.argmax(j))
    t = thr[k]
    return float(min(t, 1.0)) if np.isfinite(t) else 0.5


def run(protocol, meta, fit_predict, configs, seeds, log=print, on_fold=None):
    """meta: retained table indexed by rid. fit_predict(cfg, tr, te, seed) -> scores for te.
    Returns a DataFrame of outer-test predictions and a DataFrame of selection records."""
    y = meta["y"].values
    preds, sel = [], []
    for k, tr, te in L.folds(protocol):
        check_split(meta, tr, te, protocol)
        if on_fold:
            on_fold(protocol, k)
        inn = inner_splits(protocol, k, tr, meta)
        scores = []
        oof_by_cfg = {}
        for ci, cfg in enumerate(configs):
            oof = np.full(len(meta), np.nan)
            for a, b in inn:
                check_split(meta, a, b, protocol)
                oof[b] = fit_predict(cfg, a, b, seeds[0])
            idx = np.concatenate([b for _, b in inn])
            auc = roc_auc_score(y[idx], oof[idx]) if len(np.unique(y[idx])) > 1 else np.nan
            scores.append(auc)
            oof_by_cfg[ci] = (idx, oof[idx])
        best = int(np.nanargmax(scores))
        idx, o = oof_by_cfg[best]
        thr = youden(y[idx], o)
        sel.append(dict(protocol=protocol, fold=k, best=best, config=str(configs[best]), inner_auc=scores[best],
                        threshold=thr, all_inner_auc=[round(float(s), 4) for s in scores]))
        for s in seeds:
            p = fit_predict(configs[best], tr, te, s)
            preds.append(pd.DataFrame(dict(rid=te, fold=k, seed=s, p=p, thr=thr)))
        log(f"  {protocol} fold {k}: cfg {best} inner AUC {scores[best]:.3f} thr {thr:.3f}")
    return pd.concat(preds, ignore_index=True), pd.DataFrame(sel)


def final_selection(meta, fit_predict, configs, seed, on_fold=None):
    """Selection for the locked test: LOCO over all nine development campaigns plays the inner role.
    Returns the chosen config index, its pooled LOCO AUC and the Youden threshold. No locked file is touched."""
    y = meta["y"].values
    scores, oofs = [], []
    folds = L.folds("LOCO")
    for cfg in configs:
        oof = np.full(len(meta), np.nan)
        for k, tr, te in folds:
            if on_fold:
                on_fold("LOCO", k)
            oof[te] = fit_predict(cfg, tr, te, seed)
        idx = np.concatenate([te for _, _, te in folds])
        scores.append(roc_auc_score(y[idx], oof[idx]))
        oofs.append((idx, oof[idx]))
    best = int(np.argmax(scores))
    return best, scores, youden(y[oofs[best][0]], oofs[best][1])
