"""Text-free spectrogram pictures for Figure 1 (Python 3.14, matplotlib). Reads derived/thumbs.npz (f04)."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import p5config as C

z = np.load(os.path.join(C.DERIVED, "thumbs.npz"))
for k in z.files:
    a = z[k]
    fig = plt.figure(figsize=(1.6, 1.0), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(a, origin="lower", aspect="auto", cmap="magma", vmin=np.percentile(a, 2), vmax=np.percentile(a, 99.5))
    if k.startswith("A_"):
        ax.axhspan(69.5, 127.5, color="white", alpha=0.55, lw=0)
        ax.axhline(69.5, color="white", lw=1.0, ls="--")
    ax.set_axis_off()
    fig.savefig(os.path.join(C.FIGURES, f"thumb_{k}.png"), dpi=300)
    plt.close(fig)
print("thumbs", z.files)
