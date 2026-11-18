"""Stage 11. Deep models trained from scratch, same nested protocol as every other model.

Purpose  Does AudioSet pretraining help? Baselines trained from scratch on the logger data.
All inputs stay at the native 4,096 Hz. No resampling, so no dead band.

Models (one fixed configuration each, not tuned; declared here before any run)
  CNN  log-mel input (native band, input B front end, 128 mel bins, valid frames 0..1011 average-pooled
       by 4 in time -> 253 x 128). Four blocks [Conv3x3 -> BatchNorm -> ReLU -> MaxPool2], channels
       32, 64, 128, 256, then adaptive average pool -> Linear(256,128) -> ReLU -> Dropout(0.3) -> Linear(128,1).
       AdamW lr 1e-3, weight decay 1e-2, batch 32, 30 epochs.
  LSTM 20 MFCC per frame (orthonormal DCT-II of the same log-mel, coefficients 0 to 19), 253 frames.
       Two-layer bidirectional LSTM, hidden 64, dropout 0.2, mean over time -> Linear(128,1).
       AdamW lr 1e-3, weight decay 1e-2, batch 32, 30 epochs.
  TT   (implemented, not used in the reported analysis)
       Time-Transformer, Liu et al. (2025), J Civ Struct Health Monit 15:759-775, Table 2:
       input 1-s segment (4,096 samples), 32 patches of 128 samples, linear embedding dim 128,
       class token, learned 1-D position encoding, 5 pre-norm Transformer blocks, 8 heads,
       query/key dimension 256 per head, MLP dim 256, dropout 0.1, Adam lr 3e-5, cosine annealing,
       batch 64, 100 epochs. Each 10-s file gives 10 segments with the file label. File score =
       mean of its 10 segment probabilities. Not reproduced: the VMD denoising step, because the
       paper does not report K or the penalty factor alpha.
Common   class-weighted BCE (positive weight = negatives / positives in the training side).
         Input scaling constants (log-mel and MFCC per channel; waveform global std) come from the
         training side of each fit only. No early stopping on any evaluation set.
Output   derived/pred/scratch_<model>_LOCO.csv, _sel.csv, _FINAL_sel.json
Usage    python s11_scratch.py prep
         python s11_scratch.py CNN|LSTM|TT [--nofinal] [--time]
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.fft import dct

import p5config as C
import p5lib as L
import nested as N
import astio as A

CACHE = C.CACHE                              # local cache for decoded audio
os.makedirs(CACHE, exist_ok=True)
OUT = os.path.join(C.DERIVED, "pred")
DEV = "cuda"
torch.backends.cudnn.benchmark = False      # benchmark gave no speed-up and breaks run-to-run reproducibility
torch.backends.cudnn.deterministic = True
T_VALID = 1012                                # of 1,014 valid native-band frames, divisible by 4
POOL = 4


def prep():
    r = L.retained()
    mel = np.zeros((len(r), T_VALID // POOL, 128), np.float16)
    wav = np.zeros((len(r), C.N_FULL), np.int16)
    for i, p in enumerate(r.path):
        x = A.load(p)
        wav[i] = np.round(x * 32768).astype(np.int16)
        fb = A.fbank_B(x)[:T_VALID].numpy()
        mel[i] = fb.reshape(-1, POOL, 128).mean(1)
    np.save(os.path.join(CACHE, "mel_B_pool4.npy"), mel)
    np.save(os.path.join(CACHE, "wav_int16.npy"), wav)
    print("prep", mel.shape, wav.shape)


# ------------------------------------------------------------------ models
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers, c = [], 1
        for o in (32, 64, 128, 256):
            layers += [nn.Conv2d(c, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
            c = o
        self.f = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.h = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):                      # x (B, T, F)
        return self.h(self.f(x.unsqueeze(1))).squeeze(-1)


class LSTM(nn.Module):
    def __init__(self, d=20):
        super().__init__()
        self.r = nn.LSTM(d, 64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.h = nn.Linear(128, 1)

    def forward(self, x):
        return self.h(self.r(x)[0].mean(1)).squeeze(-1)


class Block(nn.Module):
    def __init__(self, dim=128, heads=8, dim_head=256, mlp=256, drop=0.1):
        super().__init__()
        inner = heads * dim_head
        self.h, self.dh = heads, dim_head
        self.n1, self.qkv, self.o = nn.LayerNorm(dim), nn.Linear(dim, 3 * inner, bias=False), nn.Linear(inner, dim)
        self.d1 = nn.Dropout(drop)
        self.n2 = nn.LayerNorm(dim)
        self.m = nn.Sequential(nn.Linear(dim, mlp), nn.GELU(), nn.Dropout(drop), nn.Linear(mlp, dim), nn.Dropout(drop))

    def forward(self, x):
        B, T, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        a = nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.d1.p if self.training else 0.0)
        x = x + self.d1(self.o(a.transpose(1, 2).reshape(B, T, -1)))
        return x + self.m(self.n2(x))


class TimeTransformer(nn.Module):
    def __init__(self, L_in=4096, ns=32, dim=128, depth=5):
        super().__init__()
        self.ns, self.pl = ns, L_in // ns
        self.emb = nn.Linear(self.pl, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, ns + 1, dim))
        self.drop = nn.Dropout(0.1)
        self.blocks = nn.Sequential(*[Block(dim) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x):                      # x (B, 4096)
        z = self.emb(x.view(x.shape[0], self.ns, self.pl))
        z = torch.cat([self.cls.expand(len(z), -1, -1), z], 1) + self.pos
        z = self.blocks(self.drop(z))
        return self.head(z[:, 0]).squeeze(-1)


SPEC = {
    "CNN": dict(net=CNN, lr=1e-3, wd=1e-2, bs=32, ep=30, opt="adamw", cosine=False),
    "LSTM": dict(net=LSTM, lr=1e-3, wd=1e-2, bs=32, ep=30, opt="adamw", cosine=False),
    "TT": dict(net=TimeTransformer, lr=3e-5, wd=0.0, bs=64, ep=100, opt="adam", cosine=True),
}


def n_params(name):
    return sum(p.numel() for p in SPEC[name]["net"]().parameters())


# ------------------------------------------------------------------ data per fit
class Data:
    def __init__(self, name):
        self.name = name
        if name == "TT":
            self.W = np.load(os.path.join(CACHE, "wav_int16.npy"))
        else:
            M = np.load(os.path.join(CACHE, "mel_B_pool4.npy")).astype(np.float32)
            self.X = dct(M, type=2, norm="ortho", axis=2)[:, :, :20].astype(np.float32) if name == "LSTM" else M

    def tensors(self, tr, te):
        """Training-side scaling only. Returns (Xtr, ytr-expander, Xte, n segments per file)."""
        if self.name == "TT":
            w = self.W[tr].astype(np.float32) / 32768.0
            sd = float(w.std())
            seg = lambda idx: (self.W[idx].astype(np.float32) / 32768.0 / sd).reshape(-1, 4096)
            return seg(tr), seg(te), 10
        if self.name == "CNN":
            mu, sd = self.X[tr].mean(), self.X[tr].std()
        else:
            mu, sd = self.X[tr].mean((0, 1)), self.X[tr].std((0, 1))
        return (self.X[tr] - mu) / sd, (self.X[te] - mu) / sd, 1


def fitter(name, D, y):
    sp = SPEC[name]

    def fp(cfg, tr, te, seed):
        Xtr, Xte, rep = D.tensors(tr, te)
        ytr = np.repeat(y[tr], rep).astype(np.float32)
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = sp["net"]().to(DEV)
        if sp["opt"] == "adam":
            opt = torch.optim.Adam(net.parameters(), lr=sp["lr"])
        else:
            opt = torch.optim.AdamW(net.parameters(), lr=sp["lr"], weight_decay=sp["wd"])
        steps = sp["ep"] * int(np.ceil(len(Xtr) / sp["bs"]))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps) if sp["cosine"] else None
        pw = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), device=DEV)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        xt = torch.from_numpy(np.ascontiguousarray(Xtr))       # CPU; batches moved to GPU
        yt = torch.from_numpy(ytr)
        g = torch.Generator().manual_seed(seed)
        net.train()
        for _ in range(sp["ep"]):
            perm = torch.randperm(len(xt), generator=g)
            for i in range(0, len(xt), sp["bs"]):
                b = perm[i:i + sp["bs"]]
                opt.zero_grad(set_to_none=True)
                lossf(net(xt[b].to(DEV, non_blocking=True)), yt[b].to(DEV)).backward()
                opt.step()
                if sch:
                    sch.step()
        net.eval()
        out = []
        with torch.no_grad():
            xe = torch.from_numpy(np.ascontiguousarray(Xte))
            for i in range(0, len(xe), 256):
                out.append(torch.sigmoid(net(xe[i:i + 256].to(DEV))).cpu().numpy())
        p = np.concatenate(out)
        return p.reshape(-1, rep).mean(1)
    return fp


def main():
    what = sys.argv[1]
    if what == "prep":
        return prep()
    meta = pd.read_csv(os.path.join(C.DERIVED, "retained.csv")).set_index("rid", drop=False)
    y = meta["y"].values
    D = Data(what)
    fp = fitter(what, D, y)
    tag = f"scratch_{what}"
    if "--time" in sys.argv:                   # one fit on LOCO fold 0, to size the job
        k, tr, te = L.folds("LOCO")[0]
        t0 = time.time()
        p = fp({}, tr, te, 11)
        from sklearn.metrics import roc_auc_score
        print(what, "params", n_params(what), "one fit", round(time.time() - t0), "s", "fold0 AUC",
              round(roc_auc_score(y[te], p), 3) if len(np.unique(y[te])) > 1 else "n/a", flush=True)
        return
    t0 = time.time()
    if "--finalonly" not in sys.argv:          # --finalonly: LOCO outputs already exist (CNN run of 5-6 Nov stopped here)
        pr, sel = N.run("LOCO", meta, fp, [{}], C.SEEDS)
        pr.to_csv(os.path.join(OUT, f"{tag}_LOCO.csv"), index=False)
        sel.to_csv(os.path.join(OUT, f"{tag}_LOCO_sel.csv"), index=False)
        print(f"{tag} LOCO done {time.time() - t0:.0f}s", flush=True)
    if "--nofinal" in sys.argv:
        return
    best, scores, thr = N.final_selection(meta, fp, [{}], C.SEEDS[0])
    json.dump(dict(model=what, spec={k: v for k, v in SPEC[what].items() if k != "net"}, loco_auc=scores,
                   threshold=thr, params=n_params(what)),
              open(os.path.join(OUT, f"{tag}_FINAL_sel.json"), "w"), indent=1)
    print(tag, "FINAL threshold", round(thr, 3), flush=True)


if __name__ == "__main__":
    main()
