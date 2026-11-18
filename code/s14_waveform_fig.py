"""Stage 13c. Time-domain and spectral view of three files chosen by the rule in FINDINGS.md (Stage 13c).

Rows    leak (median true positive), no leak (median true negative), borderline (nearest threshold)
Columns full 10 s waveform; 0.25 s detail from the file centre; Welch PSD 20 to 2,048 Hz (1 Hz bins)
Amplitude is shown as a fraction of full scale. No site, pipe or location detail is shown.
Output  figures/fig_waveforms.png (600 dpi) and .svg; tables/waveform_examples.csv (rid, label, score, threshold)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

import p5config as C
import p5lib as L
import wave


def load(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == C.FS and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False})
INK = "#1f3a5f"


def nearest(df, target):
    d = (df.p - target).abs()
    return df.loc[d == d.min()].rid.min()


def pick():
    c = pd.read_csv(os.path.join(C.DERIVED, "pred", "collapsed", "ast_A_pool_LP_LOCO.csv"))
    c["d"] = (c.p >= c.thr).astype(int)
    tp, tn = c[(c.y == 1) & (c.d == 1)], c[(c.y == 0) & (c.d == 0)]
    gap = (c.p - c.thr).abs()
    rows = [("Leak", nearest(tp, tp.p.median())), ("No leak", nearest(tn, tn.p.median())),
            ("Borderline", c.loc[gap == gap.min()].rid.min())]
    c = c.set_index("rid")
    return [(name, int(r), int(c.y[r]), float(c.p[r]), float(c.thr[r])) for name, r in rows]


def main():
    r = L.retained()
    ex = pick()
    fig, ax = plt.subplots(3, 3, figsize=(7.2, 5.6), gridspec_kw=dict(width_ratios=[2.2, 1.2, 1.6]))
    t = np.arange(C.N_FULL) / C.FS
    ymax = 0
    sig = []
    for name, rid, y, p, thr in ex:
        x = load(r.path[rid])
        x = x - x.mean()
        sig.append(x)
        ymax = max(ymax, np.abs(x).max())
    for i, ((name, rid, y, p, thr), x) in enumerate(zip(ex, sig)):
        lab = "leak" if y == 1 else "no leak"
        ax[i, 0].plot(t[:len(x)], x, lw=0.3, color=INK)
        ax[i, 0].set_ylim(-ymax * 1.05, ymax * 1.05)
        ax[i, 0].set_ylabel("Amplitude (FS)")
        ax[i, 0].set_title(f"{name}: labelled {lab}, score {p:.2f}, threshold {thr:.2f}", loc="left", fontsize=10)
        c0 = len(x) // 2 - int(0.125 * C.FS)
        seg = slice(c0, c0 + int(0.25 * C.FS))
        ax[i, 1].plot(t[seg] - t[c0], x[seg], lw=0.6, color=INK)
        ax[i, 1].set_ylim(-ymax * 1.05, ymax * 1.05)
        f, P = welch(x, fs=C.FS, nperseg=C.FS, noverlap=C.FS // 2)
        k = (f >= 20) & (f <= C.NYQUIST)
        ax[i, 2].plot(f[k], 10 * np.log10(P[k] + 1e-20), lw=0.6, color=INK)
        ax[i, 2].set_xlim(0, C.NYQUIST)
        ax[i, 2].set_ylabel("PSD (dB re FS²/Hz)")
    for a in ax[:2].ravel():
        a.tick_params(labelbottom=False)
    ax[2, 0].set_xlabel("Time (s)"); ax[2, 1].set_xlabel("Time (s)"); ax[2, 2].set_xlabel("Frequency (Hz)")
    lo = min(a.get_ylim()[0] for a in ax[:, 2]); hi = max(a.get_ylim()[1] for a in ax[:, 2])
    for a in ax[:, 2]:
        a.set_ylim(lo, hi)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(C.FIGURES, f"fig_waveforms.{ext}"), dpi=600)
    pd.DataFrame(ex, columns=["example", "rid", "y", "score", "threshold"]).to_csv(
        os.path.join(C.TABLES, "waveform_examples.csv"), index=False)
    print(ex)


if __name__ == "__main__":
    main()
