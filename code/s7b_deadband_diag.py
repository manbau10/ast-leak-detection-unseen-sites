"""Stage 7b. Where does the dead-band information come from? (development files only, no model)

1. Per-bin AUC of each dead mel bin's mean log-mel, and its correlation with the in-band mean.
2. Pure tones at 500, 1,000, 1,900 Hz through the input-A resampler: level of the image at
   4,096 - f and of the transition-band leakage near Nyquist.
Output derived/s7b_deadband_diag.json
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

import p5config as C
import p5lib as L
import astio as A

lo, ce, hi = A.mel_edges("A")
D = np.where(lo >= C.NYQUIST)[0]
IB = np.where(hi <= C.NYQUIST)[0]
out = {}
r = pd.read_csv(os.path.join(C.DERIVED, "retained.csv"))
dev = (r.split == "dev").values
y = r.y.values[dev]
m = np.load(os.path.join(C.DERIVED, "deadband_stats.npy"))[dev][:, :len(D)]
nv = A.n_valid_frames("A")
inb = np.array([A.fbank_A(A.load(p))[:nv][:, IB].mean().item() for p in L.retained().path.values[dev]])
out["auc_inband_mean"] = roc_auc_score(y, inb)
out["corr_inband_vs_dead_mean"] = float(np.corrcoef(inb, m.mean(1))[0, 1])
out["bins"] = [dict(bin=int(b), lo_hz=float(lo[b]), hi_hz=float(hi[b]), auc=float(roc_auc_score(y, m[:, j])),
                    mean=float(m[:, j].mean()), corr_inband=float(np.corrcoef(inb, m[:, j])[0, 1]))
               for j, b in enumerate(D)]
t = np.arange(C.N_FULL) / C.FS
fr = np.fft.rfftfreq(65536, 1 / A.AST_FS)
out["tones"] = []
for f0 in (500.0, 1000.0, 1900.0):
    x = (0.3 * np.sin(2 * np.pi * f0 * t)).astype(np.float32)
    w = A._resampler(torch.from_numpy(x).unsqueeze(0))[0].numpy()
    S = np.abs(np.fft.rfft(w[16000:16000 + 65536] * np.hanning(65536))) ** 2
    lvl = lambda g: float(10 * np.log10(S[np.argmin(abs(fr - g)) - 3:np.argmin(abs(fr - g)) + 4].max() / S.max()))
    out["tones"].append(dict(f0=f0, image_hz=C.FS - f0, image_db=lvl(C.FS - f0), image2_hz=C.FS + f0,
                             image2_db=lvl(C.FS + f0)))
json.dump(out, open(os.path.join(C.DERIVED, "s7b_deadband_diag.json"), "w"), indent=1)
print({k: v for k, v in out.items() if k != "bins"})
