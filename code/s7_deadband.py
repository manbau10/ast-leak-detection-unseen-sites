"""Stage 7. Dead-band control: does the AST use the band above the 2,048 Hz logger limit?

Input A upsamples 4,096 Hz audio to 16 kHz. Mel bins whose lower edge is at or above 2,048 Hz
(the Nyquist limit of the logger) hold no measured content. Two controls:
  1. Fill. Every dead bin of the raw log-mel is set to the Kaldi log floor, log(float32 eps)
     = -15.942, before normalisation. This is the value of an exactly band-limited signal. It is
     data free and label free. The frozen AST is then re-run. If the dead band carried
     usable information, accuracy would fall.
  2. Dead only. A classifier sees only the dead bins (mean and std over valid frames of each dead
     bin). If it beats chance, the resampled dead band leaks information (resampler roll-off
     near Nyquist, level leakage) that a model could exploit.
Output  derived/emb/A_deadfill.npz  (pool, mean_in, mean_all)
        derived/deadband_stats.npy  (n, 2 x n_dead)
        derived/s7_deadband.json
Usage   python s7_deadband.py embed | stats
"""
import os
import sys
import json
import time
import numpy as np
import torch

import p5config as C
import p5lib as L
import astio as A

OUT = os.path.join(C.DERIVED, "emb")
FLOOR = float(np.log(np.finfo(np.float32).eps))
lo, ce, hi = A.mel_edges("A")
DEAD = np.where(lo >= C.NYQUIST)[0]
NV = A.n_valid_frames("A")


def main():
    what = sys.argv[1]
    r = L.retained()
    paths = r["path"].tolist()
    info = dict(floor=FLOOR, n_dead_bins=int(len(DEAD)), first_dead_bin=int(DEAD[0]),
                first_dead_lower_hz=float(lo[DEAD[0]]), valid_frames=NV)
    if what == "stats":
        S = np.zeros((len(paths), 2 * len(DEAD)), np.float32)
        for i, p in enumerate(paths):
            fb = A.fbank_A(A.load(p))[:NV, DEAD].numpy()
            S[i] = np.concatenate([fb.mean(0), fb.std(0)])
        np.save(os.path.join(C.DERIVED, "deadband_stats.npy"), S)
        info["stats_shape"] = list(S.shape)
    if what == "embed":
        import s4_ast_embed as E
        net = E.model()
        store, t0 = {}, time.time()
        dead = torch.from_numpy(DEAD)
        for i in range(0, len(paths), 8):
            specs = []
            for p in paths[i:i + 8]:
                fb = A.fbank_A(A.load(p))
                fb[:NV, dead] = FLOOR          # padded frames stay zero, as in the checkpoint front end
                specs.append(A.normalise(fb, A.AUDIOSET_MEAN, A.AUDIOSET_STD))
            e = E.embed(net, specs, "A")
            for k in ("pool", "mean_in", "mean_all"):
                store.setdefault(k, []).append(e[k])
        store = {k: np.concatenate(v).astype(np.float32) for k, v in store.items()}
        np.savez(os.path.join(OUT, "A_deadfill.npz"), **store)
        info["embed_seconds"] = round(time.time() - t0)
    p = os.path.join(C.DERIVED, "s7_deadband.json")
    old = json.load(open(p)) if os.path.exists(p) else {}
    old.update(info)
    json.dump(old, open(p, "w"), indent=1)
    print(info, flush=True)


if __name__ == "__main__":
    main()
