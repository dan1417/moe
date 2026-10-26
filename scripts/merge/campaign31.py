#!/usr/bin/env python3
"""Wait for king #31 + the CPT backup, unpermute #31, then score merge methods.

Positive mu_hat means the candidate has lower cross-entropy than king #31
(the validator's statistic). Screens use the first sequences of fresh sample tag:A.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/root/workspace/MoE")
KING_DIGEST = "1e8f3b42509355f7be4e51921c38f902b628924ddd1196fa3e6d0f5aba5ea629"
KING_RAW = Path(f"/opt/teutonic/models/{KING_DIGEST}")
K30U = Path("/opt/teutonic/hf/k30u")
CPT_DELTA = Path("/opt/teutonic/hf/cpt-backup/line_c1_step000150_ema")
CPT = Path("/opt/teutonic/hf/cpt-ema150")
KING = Path("/opt/teutonic/hf/king31u")
PERM = Path("/opt/teutonic/hf/k31_hidden_perm.json")
RECIPES = REPO / "configs/merge/k31"
LOG = Path("/opt/teutonic/logs/campaign31.log")
SCORES = Path("/opt/teutonic/logs/scores.jsonl")
REPORT = REPO / "MERGE31.md"


def log(message: str) -> None:
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + message
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as handle:
        handle.write(line + "\n")


def checkpoint_ready(root: Path) -> bool:
    index_path = root / "model.safetensors.index.json"
    if not index_path.exists() or index_path.stat().st_size < 100:
        return False
    try:
        weight_map = json.loads(index_path.read_text())["weight_map"]
    except json.JSONDecodeError:
        return False
    for shard in set(weight_map.values()):
        path = root / shard
        if not path.exists() or path.stat().st_size < 1_000_000:
            return False
    return True


def wait_for(label: str, ready, poll: float = 30.0) -> None:
    while not ready():
        log(f"waiting for {label}")
        time.sleep(poll)
    log(f"ready: {label}")


def run(cmd: list[str], **kwargs) -> None:
    log(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO, **kwargs)


def write_recipes() -> list[Path]:
    RECIPES.mkdir(parents=True, exist_ok=True)
    king, cpt, base = str(KING), str(CPT), str(K30U)
    specs: list[tuple[str, dict]] = []

    def linear(name: str, description: str, coefs: list[float], overrides: list | None = None) -> None:
        specs.append((name, {
            "label": name,
            "description": description,
            "terms": [
                {"model": king, "coef": coefs[0]},
                {"model": cpt, "coef": coefs[1]},
                {"model": base, "coef": coefs[2]},
            ],
            "overrides": overrides or [],
        }))

    def mode(name: str, description: str, mode_name: str, mode_args: dict, terms: list[tuple[str, float]]) -> None:
        specs.append((name, {
            "label": name,
            "description": description,
            "terms": [{"model": path, "coef": coef} for path, coef in terms],
            "mode": mode_name,
            "mode_args": mode_args,
        }))

    # Priority order: task-arithmetic transplants first (most likely to beat the king),
    # then structured/module variants, then classic mergekit modes, then raw baselines last.
    for alpha in (0.25, 0.5, 0.75, 1.0, 0.15, 0.35, 1.25, 1.5, -0.25):
        tag = f"{alpha:+.2f}".replace("+", "p").replace("-", "m").replace(".", "")
        linear(
            f"task_a{tag}",
            f"#31u + {alpha}*(CPT - k30u): transplant the CPT training step",
            [1.0, alpha, -alpha],
        )
    linear(
        "task_e115_a07_h07",
        "#31u + CPT step with experts x1.15, attn x0.7, head+shared x0.7 (same shape as best vs #30)",
        [1.0, 0.0, 0.0],
        overrides=[
            {"pattern": r"\.mlp\.experts\.", "coefs": [1.0, 1.15, -1.15]},
            {"pattern": r"\.self_attn\.", "coefs": [1.0, 0.7, -0.7]},
            {"pattern": r"^lm_head\.|\.shared_experts\.", "coefs": [1.0, 0.7, -0.7]},
        ],
    )
    linear(
        "task_e13_a07_h07",
        "#31u + CPT step experts x1.3, attn/head x0.7",
        [1.0, 0.0, 0.0],
        overrides=[
            {"pattern": r"\.mlp\.experts\.", "coefs": [1.0, 1.3, -1.3]},
            {"pattern": r"\.self_attn\.", "coefs": [1.0, 0.7, -0.7]},
            {"pattern": r"^lm_head\.|\.shared_experts\.", "coefs": [1.0, 0.7, -0.7]},
        ],
    )
    linear(
        "task_e10_a05_h05_a05",
        "#31u + 0.5*CPT step with attn/head half of that",
        [1.0, 0.0, 0.0],
        overrides=[
            {"pattern": r"\.mlp\.experts\.", "coefs": [1.0, 0.5, -0.5]},
            {"pattern": r"\.self_attn\.", "coefs": [1.0, 0.25, -0.25]},
            {"pattern": r"^lm_head\.|\.shared_experts\.", "coefs": [1.0, 0.25, -0.25]},
        ],
    )

    def group(name: str, description: str, pattern: str, alpha: float) -> None:
        linear(name, description, [1.0, 0.0, 0.0], overrides=[{
            "pattern": pattern,
            "coefs": [1.0, alpha, -alpha],
        }])

    group("experts_a05", "CPT step on routed experts only, alpha 0.5", r"\.mlp\.experts\.", 0.5)
    group("experts_a10", "CPT step on routed experts only, alpha 1", r"\.mlp\.experts\.", 1.0)
    group("attn_a05", "CPT step on attention only, alpha 0.5", r"\.self_attn\.", 0.5)
    group("attn_a10", "CPT step on attention only, alpha 1", r"\.self_attn\.", 1.0)
    group("headshared_a05", "CPT step on lm_head and shared experts only, alpha 0.5",
          r"^lm_head\.|\.shared_experts\.", 0.5)
    group("headshared_a10", "CPT step on lm_head and shared experts only, alpha 1",
          r"^lm_head\.|\.shared_experts\.", 1.0)
    group("nohead_a10", "CPT step everywhere except lm_head, alpha 1",
          r"^(?!lm_head\.).+", 1.0)
    group("upper_a10", "CPT step on layers 16-44 only, alpha 1",
          r"model\.layers\.(1[6-9]|[2-3]\d|4[0-4])\.", 1.0)

    for beta in (0.5, 1.0, 2.0):
        tag = str(beta).replace(".", "")
        mode(
            f"ortho_cpt_b{tag}",
            f"#31u + {beta}* component of (CPT-k30u) orthogonal to (#31u-k30u)",
            "orthogonal",
            {"beta": beta},
            [(king, 1.0), (cpt, 0.0), (base, 0.0)],
        )
        mode(
            f"sign_cpt_b{tag}",
            f"#31u + {beta}*(CPT-k30u) only where that step agrees in sign with (#31u-k30u)",
            "sign_agree",
            {"beta": beta},
            [(king, 1.0), (cpt, 0.0), (base, 0.0)],
        )

    mode("ties_d0.3_b0.5", "TIES: keep top 30% of (CPT-k30u), scale 0.5", "ties",
         {"density": 0.3, "beta": 0.5}, [(king, 1.0), (cpt, 0.0), (base, 0.0)])
    mode("ties_d0.5_b1", "TIES: keep top 50% of (CPT-k30u), scale 1", "ties",
         {"density": 0.5, "beta": 1.0}, [(king, 1.0), (cpt, 0.0), (base, 0.0)])
    mode("bread_d0.5_drop0.1", "breadcrumbs: drop smallest 50% and largest 10% of (CPT-k30u)", "ties",
         {"density": 0.5, "beta": 1.0, "top_drop": 0.1}, [(king, 1.0), (cpt, 0.0), (base, 0.0)])
    mode("dare_p0.5_b0.5", "DARE: drop 50% of (CPT-k30u), rescale, beta 0.5", "dare",
         {"drop": 0.5, "beta": 0.5}, [(king, 1.0), (cpt, 0.0), (base, 0.0)])
    mode("dare_p0.9_b1", "DARE: drop 90% of (CPT-k30u), rescale, beta 1", "dare",
         {"drop": 0.9, "beta": 1.0}, [(king, 1.0), (cpt, 0.0), (base, 0.0)])
    for t in (0.1, 0.2):
        mode(f"slerp_cpt_t{str(t).replace('.', '')}", f"SLERP(#31u, CPT, t={t})", "slerp",
             {"t": t}, [(king, 1.0), (cpt, 0.0)])
        linear(f"soup_cpt_t{str(t).replace('.', '')}", f"{1-t}*#31u + {t}*CPT", [1 - t, t, 0.0])
        linear(f"soup_k30u_t{str(t).replace('.', '')}", f"{1-t}*#31u + {t}*k30u", [1 - t, 0.0, t])

    # Baselines last (cpt_raw already scored)
    specs.append(("k30u_raw", {
        "label": "k30u_raw",
        "description": "public merge body teutonic-II-110B-merge-k30u, no blend with #31",
        "terms": [{"model": base, "coef": 1.0}],
    }))
    specs.append(("cpt_raw", {
        "label": "cpt_raw",
        "description": "CPT EMA@150 restored from teutonic-cpt-backup, no blend with #31",
        "terms": [{"model": cpt, "coef": 1.0}],
    }))

    paths = []
    for name, payload in specs:
        path = RECIPES / f"{name}.json"
        path.write_text(json.dumps(payload, indent=1) + "\n")
        paths.append(path)
    log(f"wrote {len(paths)} recipes under {RECIPES}")
    return paths


def main() -> None:
    env = os.environ.copy()
    env["TEUTONIC_EVAL_CACHE"] = "/opt/teutonic/eval_cache"
    env["TEUTONIC_MODELS_DIRS"] = "/opt/teutonic/models"
    env["PYTHONUNBUFFERED"] = "1"
    py = str(REPO / ".venv-eval/bin/python")

    wait_for("k30u", lambda: checkpoint_ready(K30U))
    if not checkpoint_ready(CPT):
        log("restoring CPT EMA@150 from the delta backup")
        run([py, "scripts/data/ckpt_delta.py", "restore",
             "--delta", str(CPT_DELTA), "--base", str(K30U), "--out", str(CPT)])
    else:
        log("CPT restore already present")

    wait_for("king #31 raw", lambda: checkpoint_ready(KING_RAW))
    if not PERM.exists():
        run([py, "scripts/merge/perm_from_embed.py",
             "--permuted", str(KING_RAW), "--reference", str(K30U), "--out", str(PERM)])
    if not checkpoint_ready(KING):
        log("unpermuting king #31 into the k30u hidden basis")
        run([py, "scripts/merge/unpermute_checkpoint.py",
             "--src", str(KING_RAW), "--perm", str(PERM), "--out", str(KING), "--workers", "8"])
    else:
        log("king31u already present")

    recipes = write_recipes()
    attn = "fa4sim"
    try:
        probe = subprocess.run(
            [py, "-c", "from transformers.utils import is_flash_attn_4_available; raise SystemExit(0 if is_flash_attn_4_available() else 1)"],
            cwd=REPO, env=env, check=False,
        )
        if probe.returncode == 0:
            attn = "flash_attention_4"
    except OSError:
        pass
    log(f"attention backend {attn}")

    # Build the fresh validator sample first (cached afterwards).
    run([py, "scripts/eval/parity_eval.py", "sample", "--sample", "tag:A"], env=env)

    # Smoke: 32 sequences, one replica, raw CPT. Confirms the pipeline fits on 8xH100.
    def smoke(attn_name: str) -> subprocess.CompletedProcess:
        cmd = [
            py, "scripts/eval/parity_eval.py", "--attn", attn_name, "--limit", "32", "--batch-size", "2",
            "duel", "--king", str(KING), "--challenger", str(RECIPES / "cpt_raw.json"), "--sample", "tag:A",
        ]
        log(f"smoke duel, 32 sequences, attn={attn_name}")
        return subprocess.run(cmd, cwd=REPO, env=env, text=True, capture_output=True)

    if os.environ.get("CAMPAIGN_SKIP_SMOKE") == "1":
        log("skipping smoke (CAMPAIGN_SKIP_SMOKE=1)")
    else:
        smoked = smoke(attn)
        if smoked.returncode != 0 and attn != "fa4sim":
            log(f"smoke failed with {attn}, falling back to fa4sim\n{smoked.stderr[-1500:]}")
            attn = "fa4sim"
            smoked = smoke(attn)
        if smoked.returncode != 0:
            log(smoked.stdout[-1500:])
            log(smoked.stderr[-2000:])
            raise SystemExit(f"smoke duel failed ({attn})")
        log("smoke ok\n" + "\n".join(smoked.stdout.splitlines()[-8:]))

    limit = os.environ.get("CAMPAIGN_LIMIT", "800")
    replicas = os.environ.get("CAMPAIGN_REPLICAS", "1")
    batch = os.environ.get("CAMPAIGN_BATCH", "8")
    done = set()
    if SCORES.exists():
        for line in SCORES.read_text().splitlines():
            try:
                done.add(json.loads(line)["name"])
            except (json.JSONDecodeError, KeyError):
                pass
    for path in recipes:
        if path.stem in done:
            log(f"SKIP scored {path.stem}")
            continue
        cmd = [
            py, "scripts/eval/parity_eval.py", "--attn", attn,
            "--limit", limit, "--batch-size", batch, "--replicas", replicas,
            "duel", "--king", str(KING), "--challenger", str(path), "--sample", "tag:A",
        ]
        log(f"SCREEN {path.name} limit={limit} replicas={replicas} batch={batch} attn={attn}")
        started = time.time()
        proc = subprocess.run(cmd, cwd=REPO, env=env, text=True, capture_output=True)
        elapsed = time.time() - started
        (LOG.parent / "last_duel.stdout").write_text(proc.stdout)
        (LOG.parent / "last_duel.stderr").write_text(proc.stderr)
        if proc.returncode != 0:
            log(f"FAIL-HARD {path.name} in {elapsed:.0f}s\n{proc.stderr[-2500:]}")
            # don't abort the whole campaign — keep screening
            with SCORES.open("a") as handle:
                handle.write(json.dumps({"name": path.stem, "seconds": round(elapsed, 1),
                                         "summary": "FAIL", "stderr_tail": proc.stderr[-500:],
                                         "recipe": str(path)}) + "\n")
            subprocess.run([py, "scripts/merge/update_31_md.py"], cwd=REPO, env=env, check=False)
            continue
        mu = ""
        sources = {}
        for line in proc.stdout.splitlines():
            if "mu_hat=" in line and "n_sequences=" in line:
                mu = line.strip()
            parts = line.split()
            if len(parts) >= 5 and parts[0] in (
                "code-reasoning", "math-reasoning", "pretrain-behaviors", "sft-reasoning", "txt360-v2"
            ):
                try:
                    sources[parts[0]] = float(parts[-1])
                except ValueError:
                    pass
        log(f"SCORE {path.stem} {elapsed:.0f}s {mu}")
        with SCORES.open("a") as handle:
            handle.write(json.dumps({
                "name": path.stem, "seconds": round(elapsed, 1), "summary": mu, "sources": sources,
                "recipe": str(path), "attn": attn, "limit": int(limit), "batch": int(batch),
            }) + "\n")
        try:
            subprocess.run([py, "scripts/merge/update_31_md.py"], cwd=REPO, env=env, check=False)
        except OSError as exc:
            log(f"update_31_md failed: {exc}")

    log("campaign screen finished")


if __name__ == "__main__":
    main()
