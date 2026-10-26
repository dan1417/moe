#!/usr/bin/env python3
"""Map a hidden-dimension-permuted MiMo checkpoint back into a reference basis.

perm.pt holds `perm` with perm[j] = reference column matching column j of the permuted
model (from embedding column matching). Every tensor is re-indexed on its hidden axis; the
operation is an exact permutation (no arithmetic), so the function is unchanged.

  unpermute_checkpoint.py --src <permuted dir> --perm k30_hidden_perm.pt --out <dir>
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CONTRACT = ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja.txt",
            "configuration_mimo_v2.py", "modeling_mimo_v2.py")
COLS = ("embed_tokens.weight", "lm_head.weight", "qkv_proj.weight", "gate_proj.weight", "up_proj.weight", "mlp.gate.weight")
ROWS = ("o_proj.weight", "down_proj.weight")
VEC = ("layernorm.weight", "norm.weight")
KEEP = ("attention_sink_bias", "e_score_correction_bias")


def remap(name: str, w: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
    if name.endswith(COLS):
        return w[:, inv].contiguous()
    if name.endswith(ROWS):
        return w[inv, :].contiguous()
    if name.endswith(VEC):
        return w[inv].contiguous()
    if name.endswith(KEEP):
        return w.contiguous()
    raise RuntimeError(f"no permutation rule for {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--perm", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if str(args.perm).endswith(".json"):
        perm = torch.tensor(json.loads(args.perm.read_text())["perm"], dtype=torch.long)
    else:
        perm = torch.load(args.perm)["perm"]
    inv = torch.argsort(perm)
    assert torch.equal(perm[inv], torch.arange(len(perm))), "perm is not a permutation"
    index = json.loads((args.src / "model.safetensors.index.json").read_text())
    files = sorted(set(index["weight_map"].values()))
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    torch.set_num_threads(4)

    def convert(filename: str) -> tuple[str, list[str]]:
        out = {}
        with safe_open(str(args.src / filename), framework="pt", device="cpu") as h:
            for name in h.keys():
                out[name] = remap(name, h.get_tensor(name), inv)
        save_file(out, str(args.out / filename), metadata={"format": "pt"})
        return filename, list(out)

    weight_map = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for filename, names in pool.map(convert, files):
            for n in names:
                weight_map[n] = filename
    assert set(weight_map) == set(index["weight_map"]), "tensor set changed"
    total = sum((args.out / f).stat().st_size for f in files)
    (args.out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, indent=2))
    for name in CONTRACT:
        shutil.copyfile(args.src / name, args.out / name)
    (args.out / "UNPERMUTED.json").write_text(json.dumps({"source": str(args.src), "perm_file": str(args.perm), "note": "hidden-dim permutation undone; function-identical to source"}, indent=1))
    print(f"wrote {args.out}: {len(weight_map)} tensors, {total / 1e9:.1f} GB in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
