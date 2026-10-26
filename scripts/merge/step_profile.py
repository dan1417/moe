"""How did the top miners make their last steps? For (new, parent) king pairs: per tensor relative change, fraction of entries
changed, size of the change in bf16 ulps (1 ulp = a minimal Adam-crossing; many ulps = large moves / merges), and the singular-
value spectrum of the delta (LoRA = a sharp knee at its rank; full fine-tuning / merging = spread)."""
import json, sys
from pathlib import Path
import torch
from safetensors import safe_open
M = {}
def T(name, t):
    p = Path(name); idx = json.load(open(p / "model.safetensors.index.json"))["weight_map"]
    with safe_open(str(p / idx[t]), "pt") as f: return f.get_tensor(t)
tens = ["model.layers.12.self_attn.o_proj.weight", "model.layers.30.self_attn.o_proj.weight", "model.layers.30.self_attn.qkv_proj.weight",
        "model.layers.20.mlp.experts.7.down_proj.weight", "model.layers.36.mlp.experts.100.gate_proj.weight",
        "model.layers.30.mlp.shared_experts.down_proj.weight", "lm_head.weight", "model.embed_tokens.weight", "model.layers.30.mlp.gate.weight"]
import sys
for new, old in [(sys.argv[1], sys.argv[2])]:
    print(f"== {new} - {old}")
    for t in tens:
        a, b = T(new, t), T(old, t)
        d = a.float() - b.float()
        ulp = (b.float().abs().clamp_min(1e-30) * 2 ** -7)                     # bf16 spacing at the old value
        nz = d != 0; frac = float(nz.float().mean())
        steps = (d.abs() / ulp)[nz]
        hist = [float(((steps > lo) & (steps <= hi)).float().mean()) if nz.any() else 0 for lo, hi in ((0, 1.5), (1.5, 2.5), (2.5, 4.5), (4.5, 16), (16, 1e9))]
        rel = float(d.norm() / b.float().norm().clamp_min(1e-30))
        line = f"  {t.replace('model.layers.', 'L'):44s} rel {rel:.1e} changed {frac * 100:5.1f}%  ulps[1,2,3-4,5-16,>16] " + " ".join(f"{h * 100:4.0f}%" for h in hist)
        if d.dim() == 2 and min(d.shape) >= 64 and d.numel() < 6e7 and frac > 0:
            s = torch.linalg.svdvals(d)
            e = (s ** 2).cumsum(0) / (s ** 2).sum()
            line += "  energy@r8/32/128 " + "/".join(f"{float(e[min(k, len(e)) - 1]) * 100:.0f}%" for k in (8, 32, 128))
        print(line, flush=True)
