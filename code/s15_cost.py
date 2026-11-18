"""Stage 15. Inference cost of each pipeline, one 10-s file, from waveform in memory to score.

Each pipeline runs in its own fresh process so peak memory is its own.
  ast_A_pool_LP                      input-A front end + frozen AST + linear probe (deep env )
  scratch_CNN, scratch_LSTM          input-B front end (+ DCT for LSTM) + network (deep env )
Weights of the deep networks are freshly initialised: latency and memory do not depend on weight values.
Timing  CPU, 1 thread and all threads. 5 warm-up runs, then 50 timed runs; median and 90th percentile.
        GPU latency for the deep pipelines is also taken (front end on CPU, network on GPU).
Memory  peak working set of the scoring process (Windows), taken after the CPU runs and before any CUDA
        start-up. Includes the Python runtime and libraries. Weight size = stored bytes.
Usage   python s15_cost.py all          (driver; needs a quiet machine)
        python s15_cost.py one NAME     (single pipeline; prints one JSON line)
"""
import os
import sys
import json
import time
import pickle
import subprocess
import numpy as np

import p5config as C

DEEP_PY = os.environ.get("P5_PYTHON", sys.executable)    # interpreter with torch and transformers
PIPES = {"ast_A_pool_LP": DEEP_PY, "scratch_CNN": DEEP_PY, "scratch_LSTM": DEEP_PY}
RID = 2146                      # the leak example of the waveform figure; any development file would do
WARM, REPS = 5, 50


def peak_wset():
    """Peak working set of this process in bytes (Windows GetProcessMemoryInfo)."""
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.GetCurrentProcess.restype = wintypes.HANDLE
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    assert psapi.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return int(c.PeakWorkingSetSize)


def timeit(fn):
    for _ in range(WARM):
        fn()
    t = []
    for _ in range(REPS):
        t0 = time.perf_counter(); fn(); t.append(time.perf_counter() - t0)
    t = np.array(t) * 1000
    return dict(median_ms=float(np.median(t)), p90_ms=float(np.percentile(t, 90)))


def set_threads(n, est=None):
    import threadpoolctl
    threadpoolctl.threadpool_limits(n)
    try:
        import torch
        torch.set_num_threads(n)
    except ImportError:
        pass
    if est is not None:
        for k in ("n_jobs", "nthread"):
            if hasattr(est, k):
                setattr(est, k, n)
        if hasattr(est, "set_params"):
            try:
                est.set_params(n_jobs=n)
            except ValueError:
                pass


def one_deep(name):
    import torch
    import astio as A
    import p5lib as L
    x = A.load(L.retained().path[RID])
    if name == "ast_A_pool_LP":
        from transformers import ASTModel
        net = ASTModel.from_pretrained(A.SNAPSHOT, attn_implementation="sdpa").eval()
        w, b = torch.randn(768), 0.0
        front = lambda: A.normalise(A.fbank_A(x), A.AUDIOSET_MEAN, A.AUDIOSET_STD).unsqueeze(0)
        head = lambda z: z.pooler_output[0] @ w + b
        trainable = 769
    else:
        import s11_scratch as S
        from scipy.fft import dct
        net = S.SPEC[name[8:]]["net"]().eval()
        lstm = name.endswith("LSTM")

        def front():
            m = A.fbank_B(x)[:S.T_VALID].numpy().reshape(-1, S.POOL, 128).mean(1)
            if lstm:
                m = dct(m, type=2, norm="ortho", axis=1)[:, :20]
            return torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32)).unsqueeze(0)
        head = lambda z: z
        trainable = None
    n_par = sum(p.numel() for p in net.parameters())
    res = dict(params_total=int(n_par), params_trained=int(trainable or n_par),
               weight_bytes=int(sum(p.numel() * p.element_size() for p in net.parameters())
                                + sum(b.numel() * b.element_size() for b in net.buffers())))
    v = front()
    with torch.no_grad():
        for n, tag in ((1, "cpu1"), (os.cpu_count(), "cpuall")):
            set_threads(n)
            res[f"{tag}_front"] = timeit(front)
            res[f"{tag}_model"] = timeit(lambda: head(net(v)))
            res[f"{tag}_total"] = timeit(lambda: head(net(front())))
        res["peak_working_set_cpu_bytes"] = peak_wset()      # before any CUDA start-up
        if torch.cuda.is_available():
            g = net.cuda()
            vg = v.cuda()
            if name == "ast_A_pool_LP":
                w_g = w.cuda()
                hg = lambda z: z.pooler_output[0] @ w_g
            else:
                hg = lambda z: z

            def run():
                out = hg(g(vg)); torch.cuda.synchronize(); return out
            res["gpu_model"] = timeit(run)
            res["gpu_peak_alloc_bytes"] = int(torch.cuda.max_memory_allocated())
    return res


def main():
    if sys.argv[1] == "one":
        name = sys.argv[2]
        res = one_deep(name)
        res.setdefault("peak_working_set_cpu_bytes", peak_wset())
        res["name"], res["cpu_count"] = name, os.cpu_count()
        print("RESULT " + json.dumps(res), flush=True)
        return
    rows = []
    for name, py in PIPES.items():
        out = subprocess.run([py, __file__, "one", name], capture_output=True, text=True,
                             env=dict(os.environ, PYTHONUTF8="1"), cwd=os.path.dirname(__file__))
        line = [l for l in out.stdout.splitlines() if l.startswith("RESULT ")]
        if not line:
            print(name, "FAILED", out.stderr[-2000:], flush=True)
            continue
        rows.append(json.loads(line[0][7:]))
        print(name, "done", flush=True)
    json.dump(rows, open(os.path.join(C.TABLES, "cost.json"), "w"), indent=1)
    import pandas as pd
    flat = pd.json_normalize(rows)
    flat.to_csv(os.path.join(C.TABLES, "cost.csv"), index=False)
    print(flat.filter(regex="name|params|trees|nodes|weight|peak|total.median|gpu_model.median").to_string())


if __name__ == "__main__":
    main()
