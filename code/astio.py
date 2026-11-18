"""AST input pipelines for this study.

Input A ("checkpoint")  the published AST front end. 4,096 Hz -> 16,000 Hz band-limited
                        resampling, Kaldi fbank at 16 kHz, 128 mel bins over 20 to 8,000 Hz,
                        25 ms Hanning window, 10 ms hop. Mel bins above 2,048 Hz carry no
                        measured content (the dead band). Reproduces ASTFeatureExtractor exactly.
Input B ("native band") no resampling. Kaldi fbank at 4,096 Hz, 128 mel bins over 20 to
                        2,048 Hz, 100 ms Hanning window (FFT 512, 8 Hz bins), 10 ms hop. The
                        time grid (998 frames, padded to 1,024) matches the checkpoint. Every
                        mel bin lies inside the measured band.
Both are normalised as (fbank - mean) / (2 std). Input A uses the AudioSet constants of the
checkpoint. Input B uses constants computed on the training side of each split only.
"""
import os
import wave
import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as K

FS = 4096
AST_FS = 16000
MAX_LEN = 1024
N_MEL = 128
AUDIOSET_MEAN = -4.2677393
AUDIOSET_STD = 4.5689974
SNAPSHOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models",
                        "models--MIT--ast-finetuned-audioset-10-10-0.4593", "snapshots",
                        "f826b80d28226b62986cc218e5cec390b1096902")

_resampler = torchaudio.transforms.Resample(FS, AST_FS)

B_FRAME_MS = 100.0
B_SHIFT_MS = 10.0
B_HIGH = FS / 2


def load(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == FS and w.getnchannels() == 1 and w.getsampwidth() == 2
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return x


def _pad(fb):
    n = fb.shape[0]
    if n < MAX_LEN:
        fb = torch.nn.functional.pad(fb, (0, 0, 0, MAX_LEN - n))
    return fb[:MAX_LEN]


def fbank_A(x):
    """x float32 in [-1, 1] at 4,096 Hz. Returns raw log-mel (1024, 128), before normalisation."""
    w = _resampler(torch.from_numpy(x).unsqueeze(0))
    fb = K.fbank(w, sample_frequency=AST_FS, window_type="hanning", num_mel_bins=N_MEL)
    return _pad(fb)


def fbank_B(x):
    w = torch.from_numpy(x).unsqueeze(0)
    fb = K.fbank(w, sample_frequency=FS, window_type="hanning", num_mel_bins=N_MEL,
                 frame_length=B_FRAME_MS, frame_shift=B_SHIFT_MS, low_freq=20.0, high_freq=B_HIGH)
    return _pad(fb)


def n_valid_frames(variant, n_samples=40960):
    if variant == "A":
        n = int(np.ceil(n_samples * AST_FS / FS))
        fl, fs_ = int(0.025 * AST_FS), int(0.010 * AST_FS)
    else:
        n = n_samples
        fl, fs_ = int(B_FRAME_MS / 1000 * FS), int(B_SHIFT_MS / 1000 * FS)
    return 1 + (n - fl) // fs_


def mel_edges(variant):
    """Lower, centre and upper frequency (Hz) of each of the 128 mel filters."""
    if variant == "A":
        lo, hi = 20.0, AST_FS / 2
    else:
        lo, hi = 20.0, B_HIGH
    mel = lambda f: 1127.0 * np.log(1 + f / 700.0)
    imel = lambda m: 700.0 * (np.exp(m / 1127.0) - 1)
    pts = np.linspace(mel(lo), mel(hi), N_MEL + 2)
    f = imel(pts)
    return f[:-2], f[1:-1], f[2:]


def normalise(fb, mean, std):
    return (fb - mean) / (2 * std)
