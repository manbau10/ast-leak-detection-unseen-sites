"""Stage 3. Engineered features and spectra at the native 4,096 Hz rate.

Every quantity lies in 0 to 2,048 Hz, the measured band. No label is used here.

Output  derived/features.csv   rid + one column per feature (count checked and printed)
        derived/psd.npy        Welch PSD, rid order, 513 bins at 4 Hz (V^2/Hz, full scale = 1)
        derived/logmel_B.npy   not here (see s4)
        derived/s3_feature_list.csv  name, group, definition

Feature groups
  T   time domain (6)    rms_dbfs, peak_dbfs, crest_factor, skewness, kurtosis, zcr
  E   envelope (5)       Hilbert envelope mean, std, coefficient of variation, modulation index
                         (std/mean of the 50 ms frame RMS), share of frames above 2x median
  S   spectral shape (8) centroid, bandwidth, rolloff85, rolloff95, flatness, entropy,
                         peak frequency, slope (dB per kHz over 20 to 2,000 Hz)
  B   band level (16)    absolute level (dB) and share of total power in 8 bands:
                         20-100, 100-200, 200-400, 400-600, 600-800, 800-1200, 1200-1600, 1600-2000 Hz
  M   MFCC (26)          13 coefficients (40 mel bands over 20 to 2,000 Hz), mean and std
Welch: Hann window, 1,024 samples, 50% overlap, 4 Hz resolution.
"""
import os
import numpy as np
import pandas as pd
from scipy import signal, stats
import librosa

import p5config as C

BANDS = [(20, 100), (100, 200), (200, 400), (400, 600), (600, 800), (800, 1200), (1200, 1600), (1600, 2000)]
EPS = 1e-20


def load(path):
    import wave
    with wave.open(path, "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    return x - x.mean()


def feats(x):
    fs = C.FS
    out = {}
    rms = np.sqrt(np.mean(x ** 2))
    out["T_rms_dbfs"] = 20 * np.log10(rms + EPS)
    out["T_peak_dbfs"] = 20 * np.log10(np.abs(x).max() + EPS)
    out["T_crest_factor"] = np.abs(x).max() / (rms + EPS)
    out["T_skewness"] = stats.skew(x)
    out["T_kurtosis"] = stats.kurtosis(x)          # excess kurtosis, 0 for Gaussian
    out["T_zcr"] = np.mean(np.abs(np.diff(np.sign(x))) > 0)
    env = np.abs(signal.hilbert(x))
    out["E_env_mean"] = env.mean()
    out["E_env_std"] = env.std()
    out["E_env_cv"] = env.std() / (env.mean() + EPS)
    fr = librosa.util.frame(x, frame_length=int(0.05 * fs), hop_length=int(0.05 * fs))
    frms = np.sqrt((fr ** 2).mean(0))
    out["E_frame_rms_modulation"] = frms.std() / (frms.mean() + EPS)
    out["E_frames_above_2x_median"] = np.mean(frms > 2 * np.median(frms))

    f, P = signal.welch(x, fs=fs, window="hann", nperseg=1024, noverlap=512, detrend=False, scaling="density")
    sel = (f >= 20) & (f <= 2000)
    fs_, Ps = f[sel], P[sel]
    w = Ps / (Ps.sum() + EPS)
    cen = (fs_ * w).sum()
    out["S_centroid_hz"] = cen
    out["S_bandwidth_hz"] = np.sqrt(((fs_ - cen) ** 2 * w).sum())
    cs = np.cumsum(w)
    out["S_rolloff85_hz"] = fs_[np.searchsorted(cs, 0.85)]
    out["S_rolloff95_hz"] = fs_[np.searchsorted(cs, 0.95)]
    out["S_flatness"] = np.exp(np.mean(np.log(Ps + EPS))) / (Ps.mean() + EPS)
    out["S_entropy"] = -(w * np.log2(w + EPS)).sum() / np.log2(len(w))
    out["S_peak_hz"] = fs_[np.argmax(Ps)]
    out["S_slope_db_per_khz"] = np.polyfit(fs_ / 1000, 10 * np.log10(Ps + EPS), 1)[0]
    df_ = f[1] - f[0]
    tot = Ps.sum() * df_
    for lo, hi in BANDS:
        b = (f >= lo) & (f < hi)
        pb = P[b].sum() * df_
        out[f"B_{lo}_{hi}_db"] = 10 * np.log10(pb + EPS)
        out[f"B_{lo}_{hi}_share"] = pb / (tot + EPS)
    mf = librosa.feature.mfcc(y=x.astype(np.float32), sr=fs, n_mfcc=13, n_fft=1024, hop_length=256,
                              n_mels=40, fmin=20, fmax=2000)
    for i in range(13):
        out[f"M_mfcc{i}_mean"] = mf[i].mean()
        out[f"M_mfcc{i}_std"] = mf[i].std()
    return out, P.astype(np.float32), f


def main():
    r = pd.read_csv(os.path.join(C.DERIVED, "retained.csv"))
    pm = pd.read_csv(os.path.join(C.DERIVED, "manifest_private.csv"))
    pm = pm.loc[pm["retained"], ["md5", "folder", "file"]]
    assert pm["md5"].is_unique
    r = r.merge(pm, on="md5", how="left")
    assert r["file"].notna().all() and r["rid"].is_unique and len(r) == 2442
    r = r.sort_values("rid")
    rows, psd = [], []
    for t in r.itertuples():
        o, P, f = feats(load(os.path.join(C.DATA, t.folder, t.file)))
        o["rid"] = t.rid
        rows.append(o)
        psd.append(P)
    F = pd.DataFrame(rows)
    F = F[["rid"] + [c for c in F.columns if c != "rid"]]
    assert np.isfinite(F.drop(columns="rid").values).all()
    F.to_csv(os.path.join(C.DERIVED, "features.csv"), index=False)
    np.save(os.path.join(C.DERIVED, "psd.npy"), np.stack(psd))
    np.save(os.path.join(C.DERIVED, "psd_freqs.npy"), f)
    names = [c for c in F.columns if c != "rid"]
    grp = pd.Series([n.split("_")[0] for n in names]).value_counts().to_dict()
    pd.DataFrame(dict(name=names, group=[n.split("_")[0] for n in names])).to_csv(
        os.path.join(C.DERIVED, "s3_feature_list.csv"), index=False)
    print("files", len(F), "features", len(names), "by group", grp)
    print("psd", np.stack(psd).shape, "freq res", f[1] - f[0], "max f", f[-1])


if __name__ == "__main__":
    main()
