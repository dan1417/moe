#!/usr/bin/env python3
"""While public_merge scores on the GPUs, prepare training on CPU/disk only.

  - Materialize train_inits.json recipes under /data/teutonic/submit/train_<comp>_<label>
  - Link /data/teutonic/ckpt/start_<label> -> submit dir (train.sh entry point)
  - Build competition val + CPT pools when missing (no GPU; may take a while)

Does not stop public_merge or campaign43.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = Path("/data/teutonic")
INIT = D / "search" / "train_inits.json"
POOLS = D / "pools"
LOG = D / "logs" / "prepare_training.log"
MIN_MU = 1.5e-4


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def ready_ckpt(out: Path) -> bool:
    return (out / "model.safetensors.index.json").exists() and (
        (out / "SAVED").exists() or len(list(out.glob("model-*.safetensors"))) >= 40
    )


def write_merge(recipe: Path, out: Path) -> bool:
    if ready_ckpt(out):
        (out / "SAVED").touch(exist_ok=True)
        return True
    force = "--force " if out.exists() and not ready_ckpt(out) else ""
    out.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"cd {REPO} && source {D}/env.sh && ionice -c 3 nice -n 19 "
        f"python scripts/merge/write_merge.py {force}--recipe {recipe} --out {out} --gpus cpu --workers 16"
    )
    log(f"write_merge {out.name} from {recipe.name}")
    p = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
    if p.returncode != 0:
        log(f"write_merge FAILED {out.name}: {(p.stderr or p.stdout)[-800:]}")
        return False
    (out / "SAVED").touch()
    log(f"write_merge OK {out.name}")
    return True


def link_start(label: str, out: Path) -> None:
    link = D / "ckpt" / f"start_{label}"
    if link.exists() and not link.is_symlink():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() != out.resolve():
        link.unlink()
    if not link.exists():
        link.symlink_to(out.resolve())
        log(f"ckpt/start_{label} -> {out}")


def pool(name: str, comp: str, n: int, block: int, exclude: list[str]) -> None:
    dest = POOLS / name
    if (dest / "meta.json").exists():
        return
    POOLS.mkdir(parents=True, exist_ok=True)
    ex = " ".join(f"--exclude-pool {POOLS}/{e}" for e in exclude)
    cmd = (
        f"cd {REPO} && source {D}/env.sh && ionice -c 3 nice -n 19 "
        f"python scripts/data/build_diverse_pool.py --competition {comp} --n {n} --block {block} "
        f"--tag {name} --out {dest} {ex}"
    )
    log(f"building pool {name} ({comp}, n={n})")
    subprocess.run(["bash", "-c", cmd], check=False)


def prep_pools(comp: str, steps: int = 200, seqs: int = 4096) -> None:
    val = f"{comp}_val_v1"
    pool(val, comp, 2048, 1, [])
    need = max(1, (steps * seqs + 199_999) // 200_000)
    ex = [val]
    for v in range(1, need + 1):
        name = f"{comp}_cpt_v{v}"
        pool(name, comp, 200_000, 32, ex)
        ex.append(name)


def one_pass() -> None:
    if not INIT.exists():
        log("waiting for train_inits.json")
        return
    try:
        data = json.loads(INIT.read_text())
    except json.JSONDecodeError:
        log("train_inits.json invalid")
        return
    train = data.get("train_init") or {}
    written = data.get("written") or {}
    plan = {"updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "checkpoints": {}, "train_sh": {}}
    for comp in ("code", "math", "main", "text"):
        t = train.get(comp)
        if not t or (t.get("mu") or 0) < MIN_MU:
            continue
        label = t["label"]
        recipe = Path(t["recipe"])
        out = D / "submit" / f"train_{comp}_{label}"
        if recipe.exists() and write_merge(recipe, out):
            link_start(label, out)
            plan["checkpoints"][comp] = {"label": label, "path": str(out), "mu": t.get("mu")}
            plan["train_sh"][comp] = f"bash train.sh {out} {comp}"
    # pools: prioritize the competition with the strongest screen mu
    ranked = sorted(
        ((c, train[c].get("mu", 0)) for c in train if train[c].get("mu", 0) >= MIN_MU),
        key=lambda x: -x[1],
    )
    for comp, _ in ranked[:2]:
        if comp == "text":
            pool("text_val_v1", "text", 2048, 1, [])
        else:
            prep_pools(comp)
    ready_path = D / "search" / "train_ready.json"
    ready_path.write_text(json.dumps(plan, indent=1))
    log(f"train_ready.json updated ({len(plan['checkpoints'])} ckpts)")


def main() -> None:
    log("prepare_training started (CPU/disk only; merge GPUs untouched)")
    while True:
        try:
            one_pass()
        except Exception as exc:
            log(f"error: {exc}")
        time.sleep(180)


if __name__ == "__main__":
    main()
