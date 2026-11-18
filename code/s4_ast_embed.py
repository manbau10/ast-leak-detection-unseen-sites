"""Stage 4. Frozen AST embeddings for input A and input B.

No label is used. Weights are never updated here.

Input A  one pass. AudioSet normalisation constants (fixed, data free).
Input B  one pass per normalisation set. Constants are the mean and std of the raw
         native-band log-mel over valid frames of the TRAINING files of that split only:
           LOCO outer folds 0 to 8 (training = the other eight dev campaigns)
           FINAL (training = all nine dev campaigns)
Stored per file (float32, rid order), shape (n, 768):
  pool      the checkpoint's pooled output, mean of final CLS and distillation states
  mean_all  mean of final patch tokens over all rows, valid columns
  mean_in   mean over in-band rows only (A rows 0 to 5; for B all rows are in band)
  max_in    elementwise max over the same in-band patch tokens
  mean_dead (A only) mean over rows 7 to 11, the dead band. Used only for the control test.
  layer_k   mean of CLS and distillation states at the output of encoder layer k (k = 1..12),
            before the final LayerNorm (A, and B FINAL only)
Precision: fp32 throughout (fp16 was 4x slower on the GTX 1660 SUPER, measured 4 Nov 2026).
"""
import os
import sys
import json
import time
import numpy as np
import torch
from transformers import ASTModel

import p5config as C
import p5lib as L
import astio as A

DEV = "cuda"
OUT = os.path.join(C.DERIVED, "emb")
os.makedirs(OUT, exist_ok=True)


def model():
    m = ASTModel.from_pretrained(A.SNAPSHOT, attn_implementation="sdpa").to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def embed(net, specs, variant, layers=False, fp16=False):
    if variant == "A":
        rows_in, cols = L.A_INBAND_ROWS, L.A_VALID_COLS
    else:
        rows_in, cols = list(range(12)), L.B_VALID_COLS
    i_all = torch.tensor(L.patch_index(range(12), cols), device=DEV)
    i_in = torch.tensor(L.patch_index(rows_in, cols), device=DEV)
    i_dead = torch.tensor(L.patch_index(L.A_DEAD_ROWS, cols), device=DEV)
    x = torch.stack(specs).to(DEV)
    with torch.autocast("cuda", dtype=torch.float16, enabled=fp16):
        o = net(x, output_hidden_states=layers)
    h = o.last_hidden_state.float()
    res = dict(pool=o.pooler_output.float(), mean_all=h[:, i_all].mean(1), mean_in=h[:, i_in].mean(1),
               max_in=h[:, i_in].max(1).values)
    if variant == "A":
        res["mean_dead"] = h[:, i_dead].mean(1)
    if layers:
        for k in range(1, 13):
            hs = o.hidden_states[k].float()
            res[f"layer_{k}"] = (hs[:, 0] + hs[:, 1]) / 2
    return {k: v.cpu().numpy() for k, v in res.items()}


def run(net, paths, variant, mean, std, tag, layers, bs=8):
    store = {}
    t0 = time.time()
    for i in range(0, len(paths), bs):
        specs = []
        for p in paths[i:i + bs]:
            x = A.load(p)
            fb = A.fbank_A(x) if variant == "A" else A.fbank_B(x)
            specs.append(A.normalise(fb, mean, std))
        e = embed(net, specs, variant, layers)
        for k, v in e.items():
            store.setdefault(k, []).append(v)
    store = {k: np.concatenate(v).astype(np.float32) for k, v in store.items()}
    np.savez(os.path.join(OUT, f"{tag}.npz"), **store)
    print(tag, {k: v.shape for k, v in list(store.items())[:2]}, f"{time.time() - t0:.0f}s", flush=True)
    return store


def b_stats(paths):
    """Per-file sum, sum of squares and count of valid native-band log-mel values."""
    nv = A.n_valid_frames("B")
    s = np.zeros((len(paths), 3))
    for i, p in enumerate(paths):
        fb = A.fbank_B(A.load(p))[:nv].double().numpy()
        s[i] = fb.sum(), (fb ** 2).sum(), fb.size
    return s


def mean_std(s, idx):
    t = s[idx].sum(0)
    mu = t[0] / t[2]
    return float(mu), float(np.sqrt(t[1] / t[2] - mu ** 2))


def main():
    r = L.retained()
    paths = r["path"].tolist()
    net = model()
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    meta = {}
    mp = os.path.join(OUT, "meta.json")
    if os.path.exists(mp):
        meta = json.load(open(mp))

    if what == "check":
        specs = [A.normalise(A.fbank_A(A.load(p)), A.AUDIOSET_MEAN, A.AUDIOSET_STD) for p in paths[:16]]
        e16 = embed(net, specs, "A", fp16=True)["pool"]
        e32 = embed(net, specs, "A", fp16=False)["pool"]
        cos = (e16 * e32).sum(1) / np.linalg.norm(e16, axis=1) / np.linalg.norm(e32, axis=1)
        meta["fp16_vs_fp32"] = dict(max_abs=float(np.abs(e16 - e32).max()), min_cosine=float(cos.min()))
        print("fp16 check", meta["fp16_vs_fp32"], flush=True)

    if what in ("A", "all"):
        run(net, paths, "A", A.AUDIOSET_MEAN, A.AUDIOSET_STD, "A_frozen", layers=True)
        meta["A_frozen"] = dict(mean=A.AUDIOSET_MEAN, std=A.AUDIOSET_STD, source="checkpoint")

    if what in ("B", "all"):
        sp = os.path.join(OUT, "B_stats.npy")
        if not os.path.exists(sp):
            np.save(sp, b_stats(paths))
        s = np.load(sp)
        sets = [(f"B_LOCO{k}", tr) for k, tr, te in L.folds("LOCO")]
        sets.append(("B_FINAL", r.loc[r["split"] == "dev", "rid"].values))
        for tag, tr in sets:
            if os.path.exists(os.path.join(OUT, f"{tag}.npz")):
                continue
            mu, sd = mean_std(s, tr)
            run(net, paths, "B", mu, sd, tag, layers=(tag == "B_FINAL"))
            meta[tag] = dict(mean=mu, std=sd, n_train_files=int(len(tr)))
            json.dump(meta, open(mp, "w"), indent=1)
    json.dump(meta, open(mp, "w"), indent=1)


if __name__ == "__main__":
    main()
