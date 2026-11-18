"""Figure data. Collects every number plotted in the figure deck into figures/figdata.json.

Each block names its source table or file. Nothing here fits a model. Development files only,
except the 'locked' block (Stage 16 results) and the dataset counts.
Run in deep env  (needs torchaudio for the per-bin input-A spectrum).
"""
import os
import json
import wave
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.metrics import roc_auc_score

import p5config as C
import p5lib as L

T, D = C.TABLES, C.DERIVED
COL = os.path.join(D, "pred", "collapsed")
DL = {"ast_A_pool_LP": "AST", "scratch_CNN": "CNN", "scratch_LSTM": "LSTM"}


def rows(df, tags, cols=("auc", "auc_lo", "auc_hi", "balanced_accuracy", "balanced_accuracy_lo", "balanced_accuracy_hi",
                         "recall", "specificity")):
    d = df.set_index("tag")
    return {t: {c: float(d.loc[t, c]) for c in cols} for t in tags}


def per_site(meta):
    out = {}
    for tag, name in DL.items():
        c = pd.read_csv(os.path.join(COL, f"{tag}_LOCO.csv")).drop(columns=["y", "deployment", "campaign"], errors="ignore")
        c = c.merge(meta[["rid", "y", "campaign"]], on="rid")
        out[name] = {int(s): (float(roc_auc_score(g.y, g.p)) if g.y.nunique() > 1 else None) for s, g in c.groupby("campaign")}
    return out


def perbin_spectrum(meta):
    """Mean input-A log-mel per bin per development file, then per-bin AUC and class means."""
    import astio as A
    r = L.retained()
    dev = r[r.split == "dev"]
    nv = A.n_valid_frames("A")
    M = np.stack([A.fbank_A(A.load(p))[:nv].mean(0).numpy() for p in dev.path])
    y = dev.y.values
    auc = [float(roc_auc_score(y, M[:, j])) for j in range(M.shape[1])]
    lo, ce, hi = A.mel_edges("A")
    return dict(centre_hz=ce.tolist(), auc=auc, mean_leak=M[y == 1].mean(0).tolist(), mean_noleak=M[y == 0].mean(0).tolist())


def waveforms():
    ex = pd.read_csv(os.path.join(T, "waveform_examples.csv"))
    r = L.retained()
    out = []
    for _, e in ex.iterrows():
        with wave.open(r.path[e.rid], "rb") as w:
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
        x = x - x.mean()
        blk = x[: len(x) // 164 * 164].reshape(-1, 164)          # 250 blocks of 40 ms
        f, P = welch(x, fs=C.FS, nperseg=C.FS // 2, noverlap=C.FS // 4)  # 8 Hz bins
        k = (f >= 20) & (f <= C.NYQUIST)
        out.append(dict(example=e.example, y=int(e.y), score=float(e.score), thr=float(e.threshold),
                        t=(np.arange(blk.shape[0]) * 164 / C.FS).tolist(), env_max=blk.max(1).tolist(),
                        env_min=blk.min(1).tolist(), f=f[k].tolist(), psd_db=(10 * np.log10(P[k] + 1e-20)).tolist()))
    return out


def main():
    meta = pd.read_csv(os.path.join(D, "retained.csv"))
    dev = pd.read_csv(os.path.join(T, "dev_results.csv"))
    loco = dev[dev.protocol == "LOCO"]
    out = {}
    # dataset per site
    g = meta.groupby("campaign").agg(files=("rid", "size"), leak=("y", "sum"), deployments=("deployment", "nunique"),
                                     split=("split", "first"))
    out["dataset"] = {int(k): dict(files=int(v.files), leak=int(v.leak), no_leak=int(v.files - v.leak),
                                   deployments=int(v.deployments), split=v.split) for k, v in g.iterrows()}
    # main LOCO comparison and AST variants
    tags = ["ast_A_pool_LP", "ast_A_pool_MLP", "ast_A_pool_DEEP", "ast_B_pool_LP", "ast_Afill_pool_LP",
            "ast_A_mean_in_LP", "ast_A_max_in_LP", "ast_A_mean_all_LP", "ast_A_mean_dead_LP", "ast_B_mean_in_LP",
            "ast_A_mean_in_MLP", "ast_A_mean_in_DEEP", "ast_B_pool_MLP", "ast_B_pool_DEEP", "ast_DEADSTAT_LP",
            "scratch_CNN", "scratch_LSTM"]
    out["loco"] = rows(loco, tags)
    out["layers"] = rows(loco, [f"ast_A_layer{k}_LP" for k in range(1, 13)], ("auc", "auc_lo", "auc_hi"))
    out["loro"] = rows(dev[dev.protocol == "LORO"], ["ast_A_pool_LP"], ("auc", "auc_lo", "auc_hi"))
    p = pd.read_csv(os.path.join(T, "paired_LOCO.csv"))
    keep = [("ast_A_pool_LP", "scratch_CNN"), ("ast_A_pool_LP", "scratch_LSTM"), ("scratch_CNN", "scratch_LSTM"),
            ("ast_B_pool_LP", "scratch_CNN"), ("ast_Afill_pool_LP", "scratch_CNN"), ("ast_A_pool_LP", "ast_Afill_pool_LP"),
            ("ast_A_pool_LP", "ast_B_pool_LP"), ("ast_A_pool_LP", "ast_A_pool_MLP"), ("ast_A_pool_LP", "ast_A_pool_DEEP")]
    out["paired"] = [dict(ref=a, other=b, **{k: float(p[(p.ref == a) & (p.other == b)][k].iloc[0])
                                              for k in ("d_auc", "d_auc_lo", "d_auc_hi", "d_ba", "d_ba_lo", "d_ba_hi")})
                     for a, b in keep]
    out["per_site"] = per_site(meta)
    out["perbin"] = perbin_spectrum(meta)
    # integrated gradients
    out["ig_contrast"] = json.load(open(os.path.join(T, "ast_ig_contrast.json")))
    z = np.load(os.path.join(D, "ast_ig.npz"))
    yy = meta.set_index("rid").loc[z["rid"], "y"].values
    out["ig_perbin"] = dict(abs_leak=np.abs(z["ig"][yy == 1]).mean(0).tolist(), abs_noleak=np.abs(z["ig"][yy == 0]).mean(0).tolist(),
                            signed_diff=(z["ig"][yy == 1].mean(0) - z["ig"][yy == 0].mean(0)).tolist())
    out["deadband"] = json.load(open(os.path.join(D, "s7b_deadband_diag.json")))
    # tiers
    out["tiers_dev"] = json.load(open(os.path.join(T, "tiers_summary.json")))
    td = pd.read_csv(os.path.join(T, "tiers_deployments.csv"))
    out["tiers_dev_deployments"] = td[["deployment", "S", "P", "y", "tier", "anomaly"]].to_dict("records")
    # locked
    lr = pd.read_csv(os.path.join(T, "locked_results.csv")).set_index("model")
    lt = ["ast_A_pool_LP", "scratch_CNN", "scratch_LSTM", "ast_B_pool_LP", "ast_Afill_pool_LP"]
    out["locked"] = {t: {c: float(lr.loc[t, c]) for c in ("auc", "auc_lo", "auc_hi", "balanced_accuracy",
                                                           "balanced_accuracy_lo", "balanced_accuracy_hi", "recall", "specificity")}
                     for t in lt}
    ls = json.load(open(os.path.join(T, "locked_summary.json")))
    out["locked_tiers"] = ls["deployments_primary"]
    out["locked_cuts"] = ls["tier_cuts"]
    out["threshold_shift"] = {t: ls["threshold_shift"][t] for t in lt}
    ld = pd.read_csv(os.path.join(T, "locked_deployments.csv"))
    out["locked_deployments"] = ld[ld.model.isin(lt)][["model", "deployment", "S", "y", "flag"]].to_dict("records")
    # acoustic effect sizes (physical features, no model)
    fe = pd.read_csv(os.path.join(T, "feature_effects.csv")).head(15)
    out["effects"] = fe[["feature", "cliffs_delta", "lo", "hi", "sites_same_sign", "sites_both_classes"]].to_dict("records")
    out["waveforms"] = waveforms()
    cost = pd.read_csv(os.path.join(T, "cost.csv")).set_index("name")
    out["cost"] = {t: dict(params=float(cost.loc[t, "params_total"]), trained=float(cost.loc[t, "params_trained"]),
                           weight_mb=float(cost.loc[t, "weight_bytes"]) / 1e6, cpu1_ms=float(cost.loc[t, "cpu1_total.median_ms"]),
                           ram_gb=float(cost.loc[t, "peak_working_set_cpu_bytes"]) / 1e9) for t in DL}
    json.dump(out, open(os.path.join(C.FIGURES, "figdata.json"), "w"), indent=1, default=float)
    print("figdata written:", list(out))


if __name__ == "__main__":
    main()
