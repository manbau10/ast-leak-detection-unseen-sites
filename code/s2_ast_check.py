"""Stage 2a. Verify the AST pipelines before any embedding is computed.

Checks, all printed and saved to derived/s2_ast_check.json
 1. checkpoint geometry from config.json (patch, strides, layers, heads, tokens)
 2. input A reproduces the Hugging Face ASTFeatureExtractor exactly
 3. valid frames, padding, patch grid for A and B
 4. frequency span of each of the 12 patch rows, and which rows lie inside 0 to 2,048 Hz
 5. dead band in A: spectral energy above 2,048 Hz after resampling, relative to below
 6. input B: no empty mel filter
 7. pooled output equals the mean of the final CLS and distillation states
"""
import os
import json
import numpy as np
import pandas as pd
import torch
from transformers import ASTFeatureExtractor, ASTModel, ASTConfig

import p5config as C
import astio as A

rep = {}
cfg = ASTConfig.from_pretrained(A.SNAPSHOT)
rep["config"] = {k: getattr(cfg, k) for k in ["patch_size", "frequency_stride", "time_stride", "max_length",
                                              "num_mel_bins", "hidden_size", "num_hidden_layers",
                                              "num_attention_heads", "intermediate_size"]}
F_OUT = (cfg.num_mel_bins - cfg.patch_size) // cfg.frequency_stride + 1
T_OUT = (cfg.max_length - cfg.patch_size) // cfg.time_stride + 1
rep["patch_grid"] = dict(freq_rows=F_OUT, time_cols=T_OUT, patches=F_OUT * T_OUT, tokens=F_OUT * T_OUT + 2)

m = pd.read_csv(os.path.join(C.DERIVED, "manifest_private.csv"))
m = m[m.retained].reset_index(drop=True)
rng = np.random.default_rng(0)
pick = m.iloc[rng.choice(len(m), 40, replace=False)]
paths = [os.path.join(C.DATA, r.folder, r.file) for r in pick.itertuples()]

# 2. A == HF extractor
fe = ASTFeatureExtractor.from_pretrained(A.SNAPSHOT)
import torchaudio
diffs = []
for p in paths[:10]:
    x = A.load(p)
    x16 = A._resampler(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy()
    hf = fe(x16, sampling_rate=16000, return_tensors="pt")["input_values"][0]
    mine = A.normalise(A.fbank_A(x), A.AUDIOSET_MEAN, A.AUDIOSET_STD)
    diffs.append(float((hf - mine).abs().max()))
rep["A_equals_hf_extractor_max_abs_diff"] = max(diffs)

# 3. frames
rep["valid_frames"] = dict(A=A.n_valid_frames("A"), B=A.n_valid_frames("B"), padded_to=A.MAX_LEN)
fbA = A.fbank_A(A.load(paths[0]))
fbB = A.fbank_B(A.load(paths[0]))
rep["fbank_shape"] = dict(A=list(fbA.shape), B=list(fbB.shape))
rep["nonzero_frames_check"] = dict(A=int((fbA.abs().sum(1) > 0).sum()), B=int((fbB.abs().sum(1) > 0).sum()))

# 4. patch rows
rows = []
for v in ("A", "B"):
    lo, ce, hi = A.mel_edges(v)
    for r in range(F_OUT):
        b0, b1 = r * cfg.frequency_stride, r * cfg.frequency_stride + cfg.patch_size - 1
        rows.append(dict(variant=v, row=r, first_bin=b0, last_bin=b1, f_low=round(float(lo[b0]), 1),
                         f_high=round(float(hi[b1]), 1), in_band=bool(hi[b1] <= C.NYQUIST + 1e-6)))
pr = pd.DataFrame(rows)
pr.to_csv(os.path.join(C.DERIVED, "s2_patch_rows.csv"), index=False)
loA, ceA, hiA = A.mel_edges("A")
rep["A_mel_bins_with_upper_edge_le_2048Hz"] = int((hiA <= C.NYQUIST).sum())
rep["A_mel_bins_with_lower_edge_ge_2048Hz"] = int((loA >= C.NYQUIST).sum())
rep["A_patch_rows_fully_in_band"] = pr[(pr.variant == "A") & pr.in_band].row.tolist()
rep["B_patch_rows_fully_in_band"] = pr[(pr.variant == "B") & pr.in_band].row.tolist()

# 5. dead band in A, measured on the resampled waveform
ratios, deadvals, livevals = [], [], []
for p in paths:
    x = A.load(p)
    x16 = A._resampler(torch.from_numpy(x).unsqueeze(0)).squeeze(0).numpy().astype(np.float64)
    X = np.abs(np.fft.rfft(x16 - x16.mean())) ** 2
    f = np.fft.rfftfreq(len(x16), 1 / 16000)
    ratios.append(10 * np.log10(X[f > 2100].sum() / X[(f > 20) & (f < 2000)].sum()))
    fb = A.fbank_A(x)[:A.n_valid_frames("A")].numpy()
    deadvals.append(fb[:, loA >= C.NYQUIST].mean())
    livevals.append(fb[:, hiA <= C.NYQUIST].mean())
rep["A_energy_above_2100Hz_rel_below_2000Hz_dB"] = dict(median=float(np.median(ratios)), max=float(np.max(ratios)))
rep["A_logmel_mean_dead_bins"] = float(np.mean(deadvals))
rep["A_logmel_mean_live_bins"] = float(np.mean(livevals))
rep["A_logmel_floor_log_float_eps"] = float(np.log(np.finfo(np.float32).eps))

# 6. B filters
nfft = 512
fft_f = np.arange(nfft // 2 + 1) * A.FS / nfft
loB, ceB, hiB = A.mel_edges("B")
cover = [int(((fft_f > l) & (fft_f < h)).sum()) for l, h in zip(loB, hiB)]
rep["B_min_fft_bins_per_mel_filter"] = int(min(cover))
rep["B_fft_resolution_Hz"] = A.FS / nfft
x = A.load(paths[0])
rep["B_fbank_has_nan_or_inf"] = bool(~torch.isfinite(A.fbank_B(x)).all())

# 7. pooled output
model = ASTModel.from_pretrained(A.SNAPSHOT).eval()
with torch.no_grad():
    inp = A.normalise(A.fbank_A(A.load(paths[1])), A.AUDIOSET_MEAN, A.AUDIOSET_STD).unsqueeze(0)
    o = model(inp)
rep["pooled_equals_mean_cls_dist"] = float((o.pooler_output - (o.last_hidden_state[:, 0] + o.last_hidden_state[:, 1]) / 2).abs().max())
rep["sequence_length"] = int(o.last_hidden_state.shape[1])
rep["n_parameters_encoder"] = int(sum(p.numel() for p in model.parameters()))

with open(os.path.join(C.DERIVED, "s2_ast_check.json"), "w") as f:
    json.dump(rep, f, indent=1)
print(json.dumps(rep, indent=1))
print(pr.to_string(index=False))
