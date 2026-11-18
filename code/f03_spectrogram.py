"""Input-A log-mel of the leak example (rid 2146), as a text-free picture for the pipeline figure.

Frequency runs upward on the mel scale (bin 0 at the bottom). Bins whose lower edge is at or above
2,048 Hz are covered by a light veil, so the band with no measured content is visible. All labels
are added in PowerPoint, so the picture holds no text.
Two steps, because torchaudio and matplotlib live in different environments:
  python f03_spectrogram.py compute   (deep env )  -> derived/spec_inputA_leak.npy
  python f03_spectrogram.py plot      (Python 3.14) -> figures/spec_inputA_leak.png
"""
import os
import sys
import numpy as np

import p5config as C

RID = 2146
NPY = os.path.join(C.DERIVED, "spec_inputA_leak.npy")


def compute():
    import p5lib as L
    import astio as A
    r = L.retained()
    nv = A.n_valid_frames("A")
    fb = A.fbank_A(A.load(r.path[RID]))[:nv].numpy().T          # (128, frames)
    lo, ce, hi = A.mel_edges("A")
    np.save(NPY, fb)
    np.save(NPY.replace(".npy", "_lo.npy"), lo)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fb = np.load(NPY)
    lo = np.load(NPY.replace(".npy", "_lo.npy"))
    first_dead = int(np.argmax(lo >= C.NYQUIST))
    fig = plt.figure(figsize=(2.6, 2.3), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(fb, origin="lower", aspect="auto", cmap="magma", vmin=np.percentile(fb, 2), vmax=np.percentile(fb, 99.5))
    ax.axhspan(first_dead - 0.5, 127.5, color="white", alpha=0.55, lw=0)
    ax.axhline(first_dead - 0.5, color="white", lw=1.2, ls="--")
    ax.set_axis_off()
    fig.savefig(os.path.join(C.FIGURES, "spec_inputA_leak.png"), dpi=300)
    print("first dead bin", first_dead)


if __name__ == "__main__":
    {"compute": compute, "plot": plot}[sys.argv[1]]()
