#!/usr/bin/env python3
"""Gradient signal vs noise from a trainer log (cpt_fsdp.py logs g_micro = |g_b| of the first micro-step, gnorm = |g_B|
of the full step, B = K b, K = --accum). Unbiased per step: |G|^2 = (K|g_B|^2 - |g_b|^2)/(K-1), n/B = (|g_b|^2 - |g_B|^2)/(K-1)
(McCandlish et al. 2018, "An Empirical Model of Large-Batch Training"). Averaged over steps with a bootstrap SE.

    python scripts/train/grad_snr.py /data/teutonic/ckpt/<run> [--from 1] [--to 10000]
"""
import argparse, json, re
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--from", dest="lo", type=int, default=1)
ap.add_argument("--to", dest="hi", type=int, default=10**9)
a = ap.parse_args()
script = next(iter(sorted(Path("/data/teutonic/logs").glob(f"run_{a.run.name}.sh"))), None)
txt = script.read_text() if script else ""
K = int(re.search(r"--accum (\d+)", txt).group(1)) if txt else 4
b = int(re.search(r"--batch (\d+)", txt).group(1)) * 8 if txt else 128
rows = [json.loads(l) for l in (a.run / "train_log.jsonl").read_text().splitlines() if l.strip()]
rows = [r for r in rows if "g_micro" in r and a.lo <= r["step"] <= a.hi and not r.get("skipped")]
if not rows:
    raise SystemExit("no g_micro rows yet")
x = np.array([r["g_micro"] ** 2 for r in rows]); y = np.array([r["gnorm"] ** 2 for r in rows])
G2 = (K * y - x) / (K - 1); nB = (x - y) / (K - 1)
rng = np.random.default_rng(0); idx = rng.integers(0, len(rows), (2000, len(rows)))
G2_bs, nB_bs = G2[idx].mean(1), nB[idx].mean(1)
pool0 = re.search(r"--pool ([^ ,]+)", txt)
seq = int(json.loads((Path(pool0.group(1)) / "meta.json").read_text()).get("seq_len", 2048)) if pool0 else 2048
tokB = K * b * seq
print(f"{a.run.name}: {len(rows)} steps ({rows[0]['step']}..{rows[-1]['step']}), K={K}, b={b} seqs, B={tokB/1e6:.2f}M tokens")
print(f"  |g_b| {np.sqrt(x.mean()):.4f}   |g_B| {np.sqrt(y.mean()):.4f}")
print(f"  |G|^2 (population gradient) = {G2.mean():+.3e}  (SE {G2_bs.std():.1e})")
print(f"  noise at B: tr(Sigma)/B = {nB.mean():.3e}  (SE {nB_bs.std():.1e})")
if G2.mean() > 0:
    print(f"  signal share of |g_B|^2 = {G2.mean() / y.mean():.1%};  B_noise = tr(Sigma)/|G|^2 = {nB.mean() * tokB / G2.mean() / 1e6:.1f}M tokens")
else:
    print("  |G|^2 <= 0 within noise: no measurable population gradient at this batch size")
