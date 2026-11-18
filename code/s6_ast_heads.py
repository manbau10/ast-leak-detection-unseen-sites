"""Stage 6. Classifier heads on frozen AST embeddings, nested, five seeds.

Heads (every layer listed, so each ablation row maps to one declared choice)
  LP     standardise -> L2 logistic regression. C chosen inside the inner folds.
  MLP    standardise -> Linear(768,256) -> GELU -> Dropout(0.2) -> Linear(256,1)
  DEEP   standardise -> [Linear(in,512) -> LayerNorm(512) -> GELU -> Dropout(0.4)] x 3
                     -> Linear(512,1). Four linear layers.
  MLP and DEEP: AdamW, weight decay 0.01, batch 64, class-weighted BCE.
  Learning rate {1e-3, 3e-4} and epochs {30, 80} chosen inside the inner folds. No early stopping
  on any evaluation set.
Views (one embedding per file)
  A_pool     checkpoint pooled output (mean of final CLS and distillation states), input A
  A_mean_all mean of all valid final patch tokens, input A
  A_mean_in  mean of in-band final patch tokens (rows 0 to 5), input A
  A_max_in   max over in-band final patch tokens, input A
  A_layerK   mean of CLS and distillation at the output of layer K, input A (LP only)
  B_pool     pooled output, input B, normalised with outer-training constants (LOCO and FINAL only)
  B_mean_in  mean of all valid final patch tokens, input B (all rows in band)
  A_mean_dead  mean of final dead-band patch tokens (rows 7 to 11), input A (control)
  Afill_pool, Afill_mean_in  input A with dead mel bins set to the log floor (control)
  DEADSTAT   mean and std of each raw dead-band log-mel bin, no AST (control)
Usage  python s6_ast_heads.py VIEW HEAD [--protocols=LOCO,...] [--device=cuda]
"""
import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import p5config as C
import nested as N

warnings.filterwarnings("ignore")
EMB = os.path.join(C.DERIVED, "emb")
OUT = os.path.join(C.DERIVED, "pred")
os.makedirs(OUT, exist_ok=True)

GRIDS = {
    "LP": [dict(C=c) for c in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1)],
    "MLP": [dict(lr=lr, ep=ep) for lr in (1e-3, 3e-4) for ep in (30, 80)],
    "DEEP": [dict(lr=lr, ep=ep) for lr in (1e-3, 3e-4) for ep in (30, 80)],
}


class Head(nn.Module):
    def __init__(self, kind, d):
        super().__init__()
        if kind == "MLP":
            self.f = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, 1))
        else:
            layers, w = [], d
            for _ in range(3):
                layers += [nn.Linear(w, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.4)]
                w = 512
            self.f = nn.Sequential(*layers, nn.Linear(512, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def n_params(kind, d=768):
    return sum(p.numel() for p in Head(kind, d).parameters())


class Views:
    """Holds the embedding matrix; for B views it swaps matrices when the outer fold changes."""

    def __init__(self, view):
        self.view = view
        self.X = None
        if view.startswith("A_"):
            z = np.load(os.path.join(EMB, "A_frozen.npz"))
            key = view[2:].replace("layer", "layer_")
            self.X = z[key].astype(np.float32)
        elif view.startswith("Afill_"):      # dead band set to the log floor (s7_deadband.py)
            self.X = np.load(os.path.join(EMB, "A_deadfill.npz"))[view[6:]].astype(np.float32)
        elif view == "DEADSTAT":             # raw dead-band log-mel statistics only (s7_deadband.py)
            self.X = np.load(os.path.join(C.DERIVED, "deadband_stats.npy")).astype(np.float32)

    def on_fold(self, protocol, k):
        if self.view.startswith("B_"):
            if protocol != "LOCO":
                raise ValueError("B views exist for LOCO and FINAL only")
            self._load(f"B_LOCO{k}")

    def final(self):
        if self.view.startswith("B_"):
            self._load("B_FINAL")

    def _load(self, tag):
        z = np.load(os.path.join(EMB, f"{tag}.npz"))
        key = {"B_pool": "pool", "B_mean_in": "mean_in"}[self.view]
        self.X = z[key].astype(np.float32)


def fitter(head, V, y, device):
    def fp(cfg, tr, te, seed):
        X = V.X
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if head == "LP":
            m = LogisticRegression(C=cfg["C"], class_weight="balanced", max_iter=5000)
            m.fit(Xtr, y[tr])
            return m.predict_proba(Xte)[:, 1]
        torch.manual_seed(seed)
        np.random.seed(seed)
        net = Head(head, X.shape[1]).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=0.01)
        pw = torch.tensor((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1), device=device, dtype=torch.float32)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        xt = torch.tensor(Xtr, device=device, dtype=torch.float32)
        yt = torch.tensor(y[tr], device=device, dtype=torch.float32)
        g = torch.Generator(device="cpu").manual_seed(seed)
        n = len(xt)
        net.train()
        for _ in range(cfg["ep"]):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, 64):
                b = perm[i:i + 64]
                opt.zero_grad()
                lossf(net(xt[b]), yt[b]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(Xte, device=device, dtype=torch.float32))).cpu().numpy()
    return fp


def main():
    view, head = sys.argv[1], sys.argv[2]
    prot = next((a.split("=")[1] for a in sys.argv[3:] if a.startswith("--protocols=")), "LOCO").split(",")
    device = next((a.split("=")[1] for a in sys.argv[3:] if a.startswith("--device=")), "cpu")
    meta = pd.read_csv(os.path.join(C.DERIVED, "retained.csv")).set_index("rid", drop=False)
    assert (meta.index.values == np.arange(len(meta))).all()
    y = meta["y"].values
    V = Views(view)
    seeds = C.SEEDS
    fp = fitter(head, V, y, device)
    tag = f"ast_{view}_{head}"
    for p in prot:
        t0 = time.time()
        pr, sel = N.run(p, meta, fp, GRIDS[head], seeds, on_fold=V.on_fold)
        pr.to_csv(os.path.join(OUT, f"{tag}_{p}.csv"), index=False)
        sel.to_csv(os.path.join(OUT, f"{tag}_{p}_sel.csv"), index=False)
        print(f"{tag} {p} done {time.time() - t0:.0f}s", flush=True)
    if "--nofinal" in sys.argv:
        return
    best, scores, thr = N.final_selection(meta, fp, GRIDS[head], seeds[0], on_fold=V.on_fold)
    json.dump(dict(view=view, head=head, best=best, config=GRIDS[head][best], loco_auc=scores, threshold=thr,
                   head_params=(n_params(head) if head != "LP" else 769)),
              open(os.path.join(OUT, f"{tag}_FINAL_sel.json"), "w"), indent=1)
    print(tag, "FINAL", GRIDS[head][best], round(thr, 3), flush=True)


if __name__ == "__main__":
    main()
