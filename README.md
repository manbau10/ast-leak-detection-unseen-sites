# Frozen pretrained audio transformer for acoustic leak detection at unseen sites

Code for the study *A frozen pretrained audio transformer for acoustic leak detection at unseen sites with
deployment-level leak evidence tiers* (Taiwo, Zayed, Liu and Yang).

The study tests whether a frozen Audio Spectrogram Transformer (AST), pretrained on AudioSet, can detect water
leaks in night recordings from acoustic noise loggers at sites that were never used for training. It compares the
AST (with a 769-weight linear probe) with a CNN and an LSTM network trained from scratch, under nested
leave-one-site-out testing. It also groups file scores into deployment-level leak evidence tiers.

## Data

The acoustic recordings belong to the Water Supplies Department of Hong Kong and are **not** included. Site folder
names identify locations, so they are kept out of this repository. The code maps each folder to an anonymous site
number through a local `code/sites.json` file (see `code/sites_example.json`). Every published output uses
"Site n" only.

Expected input: one folder per site under `P5_DATA`, holding 10-s mono WAV files at 4,096 Hz and the label
workbook (`.xlsx`) of that site, as read by `s0_manifest.py`.

## Set-up

```bash
python -m venv env
env/Scripts/activate            # Windows; use env/bin/activate elsewhere
pip install -r requirements.txt
```

Download the public checkpoint `MIT/ast-finetuned-audioset-10-10-0.4593` from Hugging Face into `models/`
(path set in `code/astio.py`). The figure scripts (`f*.py`) need `requirements-figures.txt` and Microsoft Office
(for the MathML-to-OMML style sheet used for editable equations).

Environment variables:

| Variable | Meaning |
|---|---|
| `P5_DATA` | folder with one sub-folder of WAV files per site |
| `P5_ROOT` | working folder; `derived/`, `tables/`, `figures/` and `private/` are created here |
| `P5_CACHE` | optional cache for decoded audio (default `P5_ROOT/cache`) |

## Pipeline

Run the stages in order from `code/`.

| Stage | Script | What it does |
|---|---|---|
| 0 | `s0_manifest.py` | manifest, quality control (clipping, short files, identical audio), deployments |
| 1 | `s1_folds.py`, `guard.py` | leave-one-site-out folds, inner folds and the leakage guard |
| 2 | `s2_ast_check.py` | checks of the AST input pipelines (input A, 16 kHz; input B, native band) |
| 3 | `s3_features.py` | physical acoustic descriptors and Welch spectra (for effect sizes and figures) |
| 4 | `s4_ast_embed.py` | frozen AST embeddings (pooled output, token views, every layer) |
| 6 | `s6_ast_heads.py` | linear probe, MLP and deeper heads on the frozen embeddings, nested selection |
| 7 | `s7_deadband.py`, `s7b_deadband_diag.py` | control for the band above the 2,048 Hz logger limit |
| 9 | `s9_summary.py` | metrics with deployment-cluster bootstrap intervals |
| 10 | `s10_paired.py` | paired model gaps on the same files |
| 11 | `s11_scratch.py` | CNN and LSTM trained from scratch, same nested protocol |
| 12 | `s12_tiers.py` | deployment scores, persistence and leak evidence tiers |
| 13 | `s13_features_explain.py`, `s13b_ast_attrib.py`, `s13c_ast_ig.py`, `s13d_ig_contrast.py` | effect sizes (Cliff's delta) and integrated gradients |
| 14 | `s14_waveform_fig.py` | example files chosen by a fixed rule |
| 15 | `s15_cost.py` | inference time and memory per 10-s file |
| 16 | `s16_locked.py` | single test on the locked Sites 10 to 12 (`ast`, `scratch`, then `report`) |
| F | `f01` to `f08` | figure data and the editable PowerPoint figures |

Shared modules: `p5config.py` (paths, sites, seeds), `p5lib.py`, `nested.py` (nested selection and Youden
threshold), `metrics.py` (AUC, balanced accuracy, cluster bootstrap), `astio.py` (front ends and checkpoint).

## Main settings

- AST checkpoint `MIT/ast-finetuned-audioset-10-10-0.4593`, all 86.2 M encoder weights frozen.
- Linear probe: L2 logistic regression on the standardised 768-d pooled output, C chosen inside the training sites.
- CNN (421,825 weights) and two-layer bidirectional LSTM on 20 MFCC (143,489 weights), AdamW, 30 epochs, batch 32.
- Five seeds (11, 22, 33, 44, 55); scores averaged per file.
- 95% intervals from 2,000 deployment-cluster bootstrap resamples.

## Environments used

Python 3.10 with PyTorch 2.6 (CUDA 11.8) and Transformers 5.2 for all models. Figures were built with Python 3.14,
python-pptx 1.0.2 and latex2mathml 3.81.
