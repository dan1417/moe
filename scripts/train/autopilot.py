#!/usr/bin/env python3
"""Autopilot: train one competition until a checkpoint beats a king, switching to merge mode whenever a king changes.

    python scripts/train/autopilot.py --comp math                 # start (or adopt) the lineage, run forever
    python scripts/train/autopilot.py --comp math --start /data/teutonic/ckpt/best_math_r1s80
    python scripts/train/autopilot.py --comp math --dry           # print the plan / decisions, change nothing
(train.sh <comp> runs it under pm2 as autopilot_<comp>; merge.sh <comp> asks it for a merge round.)

One process owns everything a run needs (it replaces train_guard.py + main_watch.sh + ckpt_retention.sh + chain_runs.py
for autopilot runs, and it can be restarted at any time: it adopts the live trainer from its state file).

TRAIN    one cpt_fsdp.py run per lineage segment, launched directly (no layered argument piles). Every --state-every
         steps the trainer writes its exact state (fp32 masters + 8-bit Adam + EMA), so a crash or hang resumes the run
         exactly (same run dir, same step numbering, no fresh optimizer, no re-warmup); an OOM resumes with half the
         micro-batch (same global batch). Nothing restarts the model to change the LR: CONTROL.json does it live.
LR       the controller judges the EMA validation trend (paired: same val sequences every time) over a window after each
         change: improving -> keep; flat -> raise the LR scale x1.4 (our runs were too timid: updates mostly below bf16
         resolution); worsening -> x0.6, and back to the best exact state if the loss rose clearly. A scale that made
         things worse becomes a ceiling. At the ceiling and flat = plateau -> anneal (CONTROL anneal K), screen the
         annealed weights, then (no win) go back to the pre-anneal state and continue at a higher scale.
SCREEN   every --screen-every steps: pause the trainer, score the checkpoint on ALL FOUR competitions vs their reigning
         kings (strict king resolution: scripts/box/kings.py), resume. A run trained for MATH can win CODE (t007 did).
CONFIRM  screen mu >= bar + margin on sample A -> full independent samples B and C for that competition (still paused);
         LCB > bar on both -> STAGE: random hidden permutation + verify_submission + HF backup, listed in
         /data/teutonic/search/autopilot_ready.json and the log. Submitting stays manual (one submission per hotkey).
KINGS    every 5 min: a new king in the target competition (or MAIN, --follow-main) -> once download_kings.py has it in
         our basis: stop the trainer (checkpoint + exact state), run merge4train.py (soups / transplants of our training
         delta onto the new king, "keep training ours" included), then continue from the chosen start. If "ours" wins
         the merge round the run resumes from its exact state with nothing reset.
RETAIN   keeps the newest two checkpoints, the best checkpoint per competition, staged ones and segment starts.
State: /data/teutonic/autopilot/<comp>/state.json (+ status.json for the dashboard), log: stdout (pm2 autopilot_<comp>).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/box"))
import kings as K  # noqa: E402

D = Path("/data/teutonic")
CKPT, LOGS, POOLS, SEARCH = D / "ckpt", D / "logs", D / "pools", D / "search"
AP = D / "autopilot"
PY = str(REPO / ".venv-eval/bin/python")
T1 = D / "models/95ef7a26303df937d0dbcd2bbc2fe8d54b6acab124f552416d2988ae697091d0"  # contract files (genesis-identical)
COMPS = ("main", "math", "code", "text")

# per-competition defaults (every value can be overridden on the command line). Rationale (42.md / UPDATE.md):
#  * lr 5e-6 on every group: the TEXT sweep's best rate (6e-6 rising, 1e-5 overshooting at 50 steps); 1.5e-6 / 3e-6 moved
#    the routed experts 5x less than the winners' steps and stayed flat (MATH runs: val -0.0001 in 60 steps)
#  * routers + layers 0-7 frozen, embeddings / lm_head trainable: what the winners' steps change (step_profile.py)
#  * 2 048 x 2 048 tokens per step: 4.2M tokens (gradient noise scale ~9-20M tokens; the EMA averages the rest)
#  * overhead budget at ~100 s/step (measured: 41.6k tok/s): EMA val of the whole 2 048-seq val pool every 10 steps ~3 %
#    (512 seqs take 8 s), exact state every 20 ~3 %, a 4 x 2 400 screen every 40 steps ~8 %
BASE = dict(lr=5e-6, wd=0.1, dense_frac=1.0, warmup=100, ema=0.99, min_lr_frac=0.1, steps=10000, decay_frac=0.0,
            freeze=r"\.mlp\.gate\.", eval_every=50, save_every=100, state_every=200,
            screen_every=1000, val_n=1024, val_raw=1, screen_limit=4800, confirm_limit=40000, margin=0.0002,
            scale_min=0.3, scale_max=2.4,
            # 5DALwjE4's recipe, read off their consecutive MAIN kings (docs/ANALYSIS_2026-10-14.md): one continuous
            # run snapshotted every ~30 h; per-weight displacement Gaussian and growing like sqrt(hours) (a random walk of
            # many small AdamW updates), attention norms shrinking 0.6-1 % per step (wd ~0.1), lr ~2e-6..1e-5, 10-40k
            # updates per king = small batches (0.25-1M tokens), ALL layers + embeddings + lm_head trained, routers frozen;
            # their specialist touch-ups are 5-10 such steps. Ours before Oct 14 made ~140 updates of 4.2M tokens per run
            # and stopped: 100-1000x less displacement. Hence: 0.5M-token steps (16 x 2 x 8 x 2 048 or 4 x 2 x 8 x 8 192),
            # ~10 s/step, constant lr 5e-6 (no decay: min_lr_frac 1, decay_frac 0), wd 0.1, EMA 0.99 (~100 steps = 50M
            # tokens of smoothing; val_raw 1 = the raw weights' held-out val next to the EMA's, paired), exact state every 200
            # steps, screens every 1 000 steps (~2 h) on 4 800 seqs.
            # policy "wsd": the controller never raises / lowers / anneals; only a blow-up (val > best + 0.02) rolls back.
            policy="wsd", rollback_rise=0.02,
            # 8-bit Adam states offloaded to pinned host memory between steps: all 110B trainable params as fp32 masters +
            # fp32 EMA + fp32 grads + bf16 shards already take ~190 GB/GPU; the round trip costs ~2-3 s of a ~10-s step
            combine="bmm16", ema_on_gpu=1, no_ckpt_last=0, fused_ce=1, offload_opt=1)
PLANS = {
    # 0.5M tokens per step; MAIN = the five eval sources at their proportions, half the rows 8 192-token documents
    # (our long-document lead) and half four EOS-separated 2 048-token windows (build_main_pools.sh); pools listed but
    # not built yet are skipped at launch and picked up by the next one
    # 2 048-token WINDOW rows only (Oct 14, 10:50 UTC): the 8 192-row runs (c8/c9 and text_ap_10140501) gain on the
    # long-document component (+0.002..+0.003 on the 4k-8k bucket) and erode the windows (-0.003), which are 91 % of
    # every score; the kings gain on both. Text + MAIN window pools together: all five eval sources, text-heavy.
    "main": dict(seq_len=2048, batches="16x4", pools=[f"main_cpt_v{i}" for i in range(1, 9)] + [f"text_cpt_v{i}" for i in range(1, 9)],
                 val="main_val_v1"),
    "text": dict(seq_len=2048, batches="16x4", pools=[f"text_cpt_v{i}" for i in range(1, 9)] + [f"main_cpt_v{i}" for i in range(1, 9)],
                 val="text_val_v1"),
    "math": dict(seq_len=2048, batches="16x2", pools=[f"math_cpt_v{i}" for i in range(6, 22)], val="math_val_v1"),
    "code": dict(seq_len=2048, batches="16x2", pools=[f"code_cpt_v{i}" for i in range(1, 9)], val="code_val_v1"),
}


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} {msg}", flush=True)


def sh(cmd: list[str] | str, timeout: float = 120, **kw) -> subprocess.CompletedProcess:
    """Never let a hung helper (pm2 jlist hung in D state once for 6 min) block the controller."""
    try:
        return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout, **kw)
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(cmd, 124, exc.stdout or "", f"timeout after {timeout}s")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str))
    tmp.replace(path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


# ----------------------------------------------------------------------------------------------------------- LR policy
def trend(points: list[tuple[int, float]]) -> tuple[float, float]:
    """Least-squares slope of val vs step and its t statistic (residual-based standard error)."""
    n = len(points)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return 0.0, 0.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ys)]
    se = math.sqrt(max(sum(r * r for r in resid) / max(1, n - 2), 1e-14) / sxx)
    return slope, slope / se


def lr_decision(evals: list[dict], st: dict, cfg: dict, step: int) -> tuple[str, float, str]:
    """('keep'|'raise'|'lower'|'rollback'|'plateau', new_scale, reason) from the EMA val trend since the last change.

    evals: trainer val rows {"step", "val"} (EMA weights, paired). st: the lineage's LR state {"scale", "since",
    "ceiling", "best_val", "best_step"}. Judged only once the window after the last change spans >= 40 steps (the first
    10 steps after a change are left out: a higher LR first bumps the loss before it pays off)."""
    scale, since = st["scale"], st["since"]
    # the EMA (0.98/step) lags a change by ~50 steps: leave out the first 20 steps after it, judge after >= 60
    # a rise past rollback_rise is judged at once, not after the settling window (math_ap_10131746 sat 40 steps at
    # +0.0004 above its best waiting for the window after a cut)
    vals = [e["val"] for e in evals if e.get("val") is not None and e["step"] > since]
    if vals and st.get("best_val") is not None and vals[-1] - st["best_val"] > cfg["rollback_rise"]:
        return "rollback", max(cfg["scale_min"], scale * 0.5), (f"val {vals[-1]:.5f} is {vals[-1] - st['best_val']:+.5f} above "
                                                               f"the best {st['best_val']:.5f} (step {st['best_step']})")
    if cfg.get("policy", "wsd") == "wsd":
        return "keep", scale, "wsd: the schedule decides (hold, then the scheduled cosine decay); judged by the screens"
    window = [(e["step"], e["val"]) for e in evals if e["step"] >= since + 20 and e.get("val") is not None]
    if len(window) < 4 or window[-1][0] - window[0][0] < 30 or step - since < 60:
        return "keep", scale, f"settling (window {len(window)} evals since step {since})"
    slope, t = trend(window[-6:])
    per100 = slope * 100
    if t > 2.0 and per100 > 1e-4:
        return "lower", max(cfg["scale_min"], scale * 0.6), f"val rising {per100:+.5f}/100 steps (t {t:+.1f})"
    if t < -2.0 and per100 < -1e-4:
        return "keep", scale, f"improving {per100:+.5f}/100 steps (t {t:+.1f})"
    if step >= cfg.get("decay_at", 1 << 30):
        return "keep", scale, f"flat {per100:+.5f}/100 steps in the scheduled decay: no raise"
    # flat = no signal left at this LR, not a timid LR: every MATH run (lr 1.5e-6 .. 7e-6) stayed within 1e-4 of its
    # start, and the one raise (math_ap_10131746, x1.4 at step 60) put val +0.00035 above its best in 80 steps.
    # Harvest with an anneal instead; raising stays opt-in (cfg raise_on_flat).
    ceiling = st.get("ceiling") or cfg["scale_max"]
    if cfg.get("raise_on_flat") and scale * 1.4 <= ceiling + 1e-9:
        return "raise", scale * 1.4, f"flat {per100:+.5f}/100 steps (t {t:+.1f}): raise_on_flat"
    return "plateau", scale, f"flat {per100:+.5f}/100 steps (t {t:+.1f}) at scale {scale:g}: anneal"


# ----------------------------------------------------------------------------------------------------------- staging
STAGE_SH = """#!/usr/bin/env bash
set -euo pipefail
cd {repo} && source {d}/env.sh
mkdir -p {d}/submit/private
{py} - "{ck}" "{perm}" <<'PERM'
import json, os, sys, torch
from pathlib import Path
hidden = json.loads((Path(sys.argv[1]) / "config.json").read_text())["hidden_size"]
g = torch.Generator().manual_seed(int.from_bytes(os.urandom(8), "little"))
torch.save({{"perm": torch.randperm(hidden, generator=g)}}, sys.argv[2]); os.chmod(sys.argv[2], 0o600)
PERM
nice -n 5 {py} scripts/merge/unpermute_checkpoint.py --src "{ck}" --perm "{perm}" --out "{out}" --workers 16
rm -f "{out}/UNPERMUTED.json" "{out}/SAVED"
{py} scripts/verify/verify_submission.py --model-dir "{out}" --king-dir "{t1}" | tail -4
echo STAGED {out}
nice -n 19 ionice -c2 -n7 {py} scripts/submit/hf_upload_ckpt.py --ckpt "{out}" || echo "HF backup failed (the staged dir is intact)"
echo STAGE DONE
"""


IDENTITY_FILES = ("king_info.json", "INGESTED", "UNPERMUTED.json", "PERM.json")


def strip_identity_files(ck: Path) -> None:
    """A checkpoint written from an ingested king used to carry its king_info.json (cpt_fsdp.py < 2026-10-14 copied every
    non-shard file of the init model): src/parity.weights.model_identity then files the checkpoint under the king's
    digest and the loss cache answers with the king's losses (or stores ours under its id). Remove them before scoring."""
    for name in IDENTITY_FILES:
        p = ck / name
        if p.exists():
            p.unlink()
            log(f"{ck}: removed {name} (the init model's identity file)")


def stage_checkpoint(ck: Path, comp: str, entry: dict, label: str) -> dict:
    """Submission-ready copy of `ck` in a random hidden basis (function-preserving; the permutation stays private),
    verified against the contract, then backed up to HF, in the background (pm2 stage_<label>). Listed in
    /data/teutonic/search/autopilot_ready.json. Nothing is submitted: one submission per hotkey, by hand."""
    out = D / "submit" / f"ap_{label}"
    perm = D / "submit/private" / f"ap_{label}.perm.pt"
    script = LOGS / f"stage_{label}.sh"
    script.write_text(STAGE_SH.format(repo=REPO, d=D, py=PY, ck=ck, perm=perm, out=out, t1=T1))
    script.chmod(0o755)
    sh(["pm2", "start", "bash", "--name", f"stage_{label}"[:60], "--no-autorestart", "--output", str(LOGS / f"stage_{label}.log"),
        "--error", str(LOGS / f"stage_{label}.log"), "--", str(script)], timeout=60)
    rec = {**entry, "comp": comp, "path": str(ck), "staged_dir": str(out), "perm": str(perm),
           "stage_log": str(LOGS / f"stage_{label}.log"), "t": time.time()}
    ready = read_json(SEARCH / "autopilot_ready.json", []) or []
    write_json(SEARCH / "autopilot_ready.json", ready + [rec])
    log(f"SUBMIT-READY {comp}: {ck} -> staging {out} (log {rec['stage_log']}); submit by hand")
    return rec


# ----------------------------------------------------------------------------------------------------------- the pilot
class Pilot:
    def __init__(self, args):
        self.a = args
        self.comp = args.comp
        self.cfg = {**BASE, **PLANS[self.comp]}
        for k in ("lr", "wd", "dense_frac", "warmup", "steps", "freeze", "batches", "screen_every", "save_every", "state_every",
                  "eval_every", "screen_limit", "val_n", "ema", "scale_max", "policy", "rollback_rise", "decay_frac"):
            v = getattr(args, k, None)
            if v is not None:
                self.cfg[k] = v
        if self.cfg.get("policy") == "ema" and "rollback_rise" not in vars(args):
            self.cfg["rollback_rise"] = 3e-4
        self.dir = AP / self.comp
        self.state_path = self.dir / "state.json"
        self.st = read_json(self.state_path, None) or {"comp": self.comp, "segments": [], "events": [], "screens": [],
                                                       "best": {}, "ready": [], "kings": {}, "lr": None}
        self.dry = args.dry
        self._last_king_check = 0.0
        self._seen_rows = 0
        self._evals: list[dict] = []
        self._stopped_for_screen = False

    # ---------------------------------------------------------------- bookkeeping
    def save(self) -> None:
        if not self.dry:
            write_json(self.state_path, self.st)
            self.status()

    def event(self, kind: str, **kw) -> None:
        e = {"t": time.time(), "kind": kind, **kw}
        self.st["events"] = (self.st["events"] + [e])[-400:]
        log(f"[{kind}] " + " ".join(f"{k}={v}" for k, v in kw.items()))
        self.save()

    def seg(self) -> dict | None:
        return self.st["segments"][-1] if self.st["segments"] else None

    def run_dir(self) -> Path:
        return CKPT / self.seg()["run"]

    def status(self) -> None:
        s = self.seg()
        write_json(self.dir / "status.json", {
            "t": time.time(), "comp": self.comp, "phase": self.st.get("phase"), "run": s and s["run"],
            "start": s and s["start"], "lr": self.st.get("lr"), "best": self.st.get("best"), "kings": self.st.get("kings"),
            "ready": self.st.get("ready"), "last_screen": (self.st.get("screens") or [None])[-1],
            "events": self.st["events"][-30:]})

    # ---------------------------------------------------------------- trainer process
    def trainer_pids(self) -> list[str]:
        if not self.seg():
            return []
        out = sh(["pgrep", "-f", f"scripts/train/cpt_fsdp.py .*--out {self.run_dir()}( |$)"], timeout=20).stdout
        return out.split()

    def any_trainer(self) -> list[str]:
        return sh(["pgrep", "-f", "[s]cripts/train/cpt_fsdp.py"], timeout=20).stdout.split()

    def gpu_pids(self) -> list[str]:
        return [x for x in sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader", timeout=60).stdout.split() if x.strip()]

    def free_gpus(self, why: str) -> None:
        for i in range(48):
            pids = self.gpu_pids()
            if not pids:
                return
            if i == 12:
                log(f"GPUs still busy ({why}): kill -9 {' '.join(pids)}")
                sh(["kill", "-9", *pids])
            time.sleep(5)
        raise RuntimeError(f"GPUs still busy after 4 min ({why})")

    def per_gpu(self) -> tuple[int, int]:
        b, a = (int(x) for x in self.cfg["batches"].split("x"))
        return b, a

    def write_control(self, **kw) -> None:
        """Persistent fields (lr_scale, anneal) are re-sent every time; one-shots (save/state/stop) get a new id."""
        path = self.run_dir() / "CONTROL.json"
        cur = read_json(path, {}) or {}
        cur.update({k: v for k, v in kw.items() if k in ("lr_scale", "anneal")})
        for k in ("save", "state", "stop"):
            cur[k] = bool(kw.get(k, False))
        if any(kw.get(k) for k in ("save", "state", "stop")) or "id" not in cur:
            cur["id"] = int(cur.get("id", 0)) + 1
        cur["by"] = "autopilot"
        if not self.dry:
            write_json(path, cur)

    def ensure_pools(self) -> tuple[str, str]:
        names, val = self.cfg["pools"], self.cfg["val"]
        missing = [n for n in names + [val] if not (POOLS / n / "meta.json").exists()]
        if missing and not self.dry and self.comp in ("main", "text"):
            # mixes come from scripts/box/build_main_pools.sh / build_text_pools.sh (hours of range reads, run under pm2
            # ahead of time); a launch uses whichever are on disk
            have = [n for n in names if (POOLS / n / "meta.json").exists()]
            if not have or not (POOLS / val / "meta.json").exists():
                raise RuntimeError(f"{self.comp}: no pools yet; run bash scripts/box/build_{self.comp}_pools.sh (missing {missing})")
            self.event("pools", missing=",".join(missing), using=",".join(have))
        elif missing and not self.dry:
            self.event("pools", building=",".join(missing))
            if False:
                pass
            else:
                for n in [val] + names:
                    if (POOLS / n / "meta.json").exists():
                        continue
                    block = 1 if n == val else 32
                    count = 2048 if n == val else 200000
                    excl = [str(POOLS / p) for p in [val] + names if p != n and (POOLS / p / "meta.json").exists()]
                    excl += [str(p) for p in POOLS.glob(f"{self.comp}_cpt_v*") if (p / "meta.json").exists() and str(p) not in excl]
                    cmd = [PY, "scripts/data/build_diverse_pool.py", "--competition", self.comp, "--n", str(count), "--block",
                           str(block), "--tag", n, "--out", str(POOLS / n)] + sum((["--exclude-pool", e] for e in excl), [])
                    r = sh(cmd, timeout=3600, cwd=REPO)
                    if r.returncode != 0 or not (POOLS / n / "meta.json").exists():
                        raise RuntimeError(f"pool {n} failed: {r.stderr[-400:]}")
        pools = ",".join(str(POOLS / n) for n in names if (POOLS / n / "meta.json").exists())
        if not pools:
            raise RuntimeError(f"no training pools for {self.comp}")
        return pools, str(POOLS / val)

    def _build_text_pools(self) -> None:
        def build(name, n, block, excl):
            if (POOLS / name / "meta.json").exists():
                return
            cmd = [PY, "scripts/data/build_diverse_pool.py", "--competition", "text", "--n", str(n), "--block", str(block),
                   "--tag", name, "--out", str(POOLS / name)] + sum((["--exclude-pool", str(POOLS / e)] for e in excl), [])
            r = sh(cmd, timeout=3600, cwd=REPO)
            if r.returncode != 0:
                raise RuntimeError(f"pool {name} failed: {r.stderr[-400:]}")
        build("text_val_v1", 2048, 1, [])
        wins = [f"text_cpt_v{v}" for v in range(1, 5)]
        for i, w in enumerate(wins):
            build(w, 200000, 32, ["text_val_v1"] + wins[:i])
        csv = ",".join(str(POOLS / w) for w in wins)
        if not (POOLS / "text_long_v1/meta.json").exists():
            sh([PY, "scripts/data/build_long_pool.py", "--pools", csv, "--out", str(POOLS / "text_long_v1"), "--join", "4"],
               timeout=3600, cwd=REPO)
        sh([PY, "scripts/data/build_mix_pool.py", "--long", str(POOLS / "text_long_v1"), "--windows", csv, "--out",
            str(POOLS / "text_mix_v1"), "--n", "200000"], timeout=3600, cwd=REPO)

    def launch(self, resume_state: Path | None = None) -> None:
        """(Re)start the trainer of the current segment. resume_state: an exact state of THIS run dir."""
        s = self.seg()
        out = self.run_dir()
        pools, val = self.ensure_pools()
        b, a = s.get("batch") or self.per_gpu()
        steps = s["steps"]
        args = [PY, "-m", "torch.distributed.run", "--nproc_per_node", "8", "--master_port", "29500", "scripts/train/cpt_fsdp.py",
                "--init", s["start"], "--pool", pools, "--val", val, "--out", str(out), "--steps", str(steps),
                "--batch", str(b), "--accum", str(a), "--lr", f"{self.cfg['lr']:.3e}", "--lr-dense-frac", str(self.cfg["dense_frac"]),
                "--wd", str(self.cfg.get("wd", 0.0)),
                "--warmup", str(s.get("warmup", self.cfg["warmup"])), "--min-lr-frac", str(self.cfg["min_lr_frac"]),
                "--decay-start", str(int(round(steps * (1 - self.cfg["decay_frac"])))), "--ema", str(self.cfg["ema"]), "--val-raw", str(self.cfg["val_raw"]),
                "--val-n", str(s.get("val_n", 512)), "--eval-every", str(self.cfg["eval_every"]), "--save-every", str(self.cfg["save_every"]),
                "--state-every", str(self.cfg["state_every"]), "--state-keep", "2", "--freeze", self.cfg["freeze"],
                "--dtype", "fp32", "--optim", "adamw8bit", "--offload-opt", str(self.cfg.get("offload_opt", 0)), "--attn", "flex_attention",
                "--contract-dir", str(T1), "--seed", str(s.get("seed", 0)), "--skip-seqs", str(s.get("skip", 0)),
                "--combine", self.cfg["combine"], "--ema-on-gpu", str(self.cfg["ema_on_gpu"]),
                "--no-ckpt-last", str(self.cfg["no_ckpt_last"]), "--fused-ce", str(self.cfg["fused_ce"])]
        if s.get("start_val") is not None and resume_state is None:
            args += ["--start-val", str(s["start_val"])]
        if resume_state is not None:
            args += ["--resume-state", str(resume_state)]
        script = LOGS / f"run_{s['run']}.sh"  # the dashboard reads the recipe from this file
        body = ("#!/usr/bin/env bash\n# written by scripts/train/autopilot.py\ncd " + str(REPO) + "\n"
                "export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 TEUTONIC_EVAL_CACHE=/data/teutonic/eval_cache\n"
                "export TRITON_CACHE_DIR=/data/teutonic/cache/triton TORCHINDUCTOR_CACHE_DIR=/data/teutonic/cache/inductor\n"
                "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n"
                "exec " + " ".join(_q(x) for x in args) + "\n")
        if self.dry:
            log("DRY: would launch\n" + body)
            return
        if not resume_state and out.exists() and any(out.iterdir()):
            raise RuntimeError(f"run dir {out} exists and this is not a resume")
        if resume_state is not None:  # rows of steps after the state belong to a crashed tail or an abandoned branch
            n = int(json.loads((resume_state / "STATE.json").read_text())["step"])
            p = out / "train_log.jsonl"
            if p.exists():
                keep, drop = [], []
                for line in p.read_text().splitlines():
                    try:
                        (keep if json.loads(line).get("step", 0) <= n else drop).append(line)
                    except ValueError:
                        pass
                if drop:
                    with (out / "train_log.abandoned.jsonl").open("a") as h:
                        h.write("\n".join(drop) + "\n")
                    p.write_text("\n".join(keep) + "\n")
            self._seen_rows = len(keep) if p.exists() else 0
            self._evals = [e for e in self._evals if e["step"] <= n]
        self.free_gpus("launch")
        out.mkdir(parents=True, exist_ok=True)
        script.write_text(body)
        script.chmod(0o755)
        sh(["pm2", "delete", s["run"]], timeout=60)
        r = sh(["pm2", "start", "bash", "--name", s["run"], "--no-autorestart", "--time", "--output",
                str(LOGS / f"pm2_{s['run']}.out.log"), "--error", str(LOGS / f"pm2_{s['run']}.err.log"), "--", str(script)],
               timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"pm2 start failed: {r.stderr[-300:]}")
        s["launched"] = time.time()
        s.setdefault("launches", []).append({"t": time.time(), "resume": str(resume_state) if resume_state else None,
                                             "batch": [b, a]})
        self.write_control(lr_scale=self.st["lr"]["scale"], anneal=self.st["lr"].get("anneal", 0))
        self.st["phase"] = "train"
        self.event("launch", run=s["run"], start=s["start"], resume=resume_state and resume_state.name, batch=f"{b}x{a}",
                   lr=f"{self.cfg['lr']:.1e}x{self.st['lr']['scale']:.2f}", steps=steps)
        render = LOGS / "render_dashboard.sh"
        sh(["pm2", "restart", "dashboard_render"], timeout=60) if render.exists() else None

    def new_segment(self, start: str, why: str, start_val: float | None = None, keep_lr: bool = True) -> None:
        name = f"{self.comp}_ap_{time.strftime('%m%d%H%M', time.gmtime())}"
        while (CKPT / name).exists():
            name += "x"
        b, a = self.per_gpu()
        used = self.st.get("seqs_used", 0)
        seg = {"run": name, "start": str(start), "steps": self.cfg["steps"], "why": why, "t": time.time(),
               "warmup": self.cfg["warmup"], "seed": 0, "skip": used, "per_step": b * a * 8,
               "val_n": self.cfg["val_n"],  # fixed per segment: val deltas are paired against the segment's step-0 val
               "kings": {c: (self.st["kings"].get(c) or {}).get("digest") for c in COMPS}}
        if start_val is not None:
            seg["start_val"] = start_val
        self.st["segments"].append(seg)
        old = self.st.get("lr") or {}
        self.st["lr"] = {"scale": old.get("scale", 1.0) if keep_lr else 1.0, "since": 0, "ceiling": old.get("ceiling"),
                         "best_val": None, "best_step": 0, "anneal": 0, "history": []}
        self._seen_rows, self._evals = 0, []
        self.event("segment", run=name, start=start, why=why, scale=self.st["lr"]["scale"])
        self.launch()

    # ---------------------------------------------------------------- logs
    def read_log(self) -> list[dict]:
        p = self.run_dir() / "train_log.jsonl"
        if not p.exists():
            return []
        rows = []
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        return rows

    def newest_state(self) -> Path | None:
        out = self.run_dir()
        st = [p for p in out.glob("state-*") if p.is_dir() and (p / "STATE.json").exists()]
        st.sort(key=lambda p: int(p.name.split("-")[1]))
        return st[-1] if st else None

    def saved_ckpts(self, run_dir: Path | None = None) -> list[Path]:
        out = run_dir or self.run_dir()
        found = [p for p in out.glob("step-*") if p.is_dir() and not p.is_symlink() and (p / "SAVED").exists()]
        found.sort(key=lambda p: int(p.name.split("-")[1]))
        if (out / "final" / "SAVED").exists():
            found.append(out / "final")
        return found

    def ckpt_step(self, p: Path) -> int:
        return self.seg()["steps"] if p.name == "final" else int(p.name.split("-")[1])

    # ---------------------------------------------------------------- crash / hang
    def check_health(self) -> None:
        s = self.seg()
        out = self.run_dir()
        alive = bool(self.trainer_pids())
        if (out / "final" / "SAVED").exists() and not alive:
            return self.segment_done()
        if s.get("stopping"):
            if not alive:
                s.pop("stopping")
                self.save()
            return
        if not alive:
            if time.time() - s.get("launched", 0) < 120:  # torchrun not up yet
                return
            text = _tail(LOGS / f"pm2_{s['run']}.out.log", 30000) + _tail(LOGS / f"pm2_{s['run']}.err.log", 30000)
            oom = bool(re.search(r"OutOfMemory|out of memory", text[-20000:]))
            nan = "3 non-finite steps in a row" in text[-20000:]
            return self.recover("oom" if oom else "nan" if nan else "crash")
        # hang: no log line for 3 x the usual step / val / save time and not paused on purpose
        p = out / "train_log.jsonl"
        last = max(p.stat().st_mtime if p.exists() else 0, (LOGS / f"pm2_{s['run']}.out.log").stat().st_mtime
                   if (LOGS / f"pm2_{s['run']}.out.log").exists() else 0, s.get("launched", 0), s.get("resumed_at", 0))
        steps = [r["step_s"] for r in self.read_log()[-30:] if "step_s" in r]
        typical = sorted(steps)[len(steps) // 2] if steps else 300
        limit = max(40 * 60, 3 * typical + 20 * 60)
        if (out / "PAUSE").exists():
            limit = max(limit, 3 * 3600)  # a full confirmation keeps the trainer paused for up to ~1.5 h
        if time.time() - last > limit:
            self.recover(f"hang ({(time.time() - last) / 60:.0f} min silent)")

    def recover(self, why: str) -> None:
        s = self.seg()
        s["crashes"] = [t for t in s.get("crashes", []) if time.time() - t < 6 * 3600] + [time.time()]
        if len(s["crashes"]) > 5:
            self.st["phase"] = "needs_attention"
            self.event("give_up", why=f"{len(s['crashes'])} crashes in 6 h, last: {why}")
            raise SystemExit(3)
        for pid in self.trainer_pids():
            sh(["kill", "-9", pid])
        sh(["pm2", "stop", s["run"]], timeout=60)
        (self.run_dir() / "PAUSE").unlink(missing_ok=True)
        if why == "oom":
            b, a = s.get("batch") or self.per_gpu()
            if b > 1:
                s["batch"] = [b // 2, a * 2]
        if why == "nan":
            self.st["lr"]["scale"] = max(self.cfg["scale_min"], self.st["lr"]["scale"] * 0.5)
            self.st["lr"]["ceiling"] = self.st["lr"]["scale"] / 0.5 * 0.85
        state = self.newest_state()
        self.event("recover", why=why, state=state and state.name, batch=s.get("batch"), scale=self.st["lr"]["scale"])
        if state is not None:
            return self.launch(resume_state=state)
        ck = [c for c in self.saved_ckpts() if c.name != "final"]
        if ck:  # no exact state yet: lossy restart from the newest checkpoint as a new segment (fresh optimizer)
            b, a = s.get("batch") or self.per_gpu()
            done = self.ckpt_step(ck[-1])
            self.st["segments"].append({**s, "run": s["run"] + "_r", "start": str(ck[-1]), "steps": s["steps"] - done,
                                        "skip": s.get("skip", 0) + done * b * a * 8, "warmup": 5, "crashes": [],
                                        "launches": [], "why": f"lossy restart after {why}"})
            self._seen_rows, self._evals = 0, []
            return self.launch()
        shutil.rmtree(self.run_dir(), ignore_errors=True)
        self.launch()

    # ---------------------------------------------------------------- LR control
    def control_lr(self) -> None:
        rows = self.read_log()
        new = rows[self._seen_rows:]
        self._seen_rows = len(rows)
        lr = self.st["lr"]
        for r in new:
            if "val" in r and r.get("step", 0) > 0:
                self._evals.append(r)
                if lr["best_val"] is None or r["val"] < lr["best_val"]:
                    lr["best_val"], lr["best_step"] = r["val"], r["step"]
                    lr["best_state_pending"] = r["step"]
        self.keep_best_state()
        if not new or not any("val" in r for r in new):
            return
        step = max(r.get("step", 0) for r in rows)
        if lr.get("anneal"):
            return  # an anneal runs to its end (then: screen, restore, continue)
        cyc = lr.get("cycle")
        if cyc:  # a fixed test cycle: no LR decisions, an anneal at its anneal_at step, judged at the end (judge_cycle)
            if step >= cyc["anneal_at"]:
                self.start_anneal(step, "cycle", k=cyc["k"])
            return self.save()
        req = self.dir / "REQUEST_ANNEAL"
        if req.exists():
            spec = read_json(req, {}) or {}
            req.unlink(missing_ok=True)
            lr["last"] = {"step": step, "act": "plateau", "why": "requested"}
            return self.start_anneal(step, "requested", k=int(spec.get("k", 0)) or None)
        act, scale, why = lr_decision(self._evals, lr, {**self.cfg, "decay_at": int(self.seg()["steps"] * (1 - self.cfg["decay_frac"]))}, step)
        lr["last"] = {"step": step, "act": act, "why": why}
        if act == "keep":
            self.save()
            return
        if act in ("raise", "lower"):
            if act == "lower" and lr["history"] and lr["history"][-1].get("act") == "raise":
                lr["ceiling"] = lr["scale"] * 0.999  # the scale we just raised to made it worse: never again this segment
            lr["history"].append({"step": step, "act": act, "from": lr["scale"], "to": scale, "why": why})
            lr["scale"], lr["since"] = scale, step
            self.write_control(lr_scale=scale)
            self.event("lr", act=act, scale=f"{scale:.3f}", lr=f"{self.cfg['lr'] * scale:.2e}", why=why)
        elif act == "rollback":
            best = self.state_near(lr["best_step"])
            lr["history"].append({"step": step, "act": act, "from": lr["scale"], "to": scale, "why": why, "state": best and best.name})
            lr["ceiling"] = lr["scale"] * 0.999
            lr["scale"], lr["since"] = scale, (int(best.name.split("-")[1]) if best else step)
            self.event("lr", act="rollback", scale=f"{scale:.3f}", to=best and best.name, why=why)
            if best is not None:
                self.restart_from_state(best)
            else:
                self.write_control(lr_scale=scale)
        elif act == "plateau":
            self.start_anneal(step, why)

    def keep_best_state(self) -> None:
        """Hard-link the newest exact state at or before the best val step as best_state-N (free; the trainer keeps only
        its newest two states, a rollback needs the best one)."""
        lr = self.st["lr"]
        pend = lr.get("best_state_pending")
        if pend is None:
            return
        cands = [p for p in self.run_dir().glob("state-*") if (p / "STATE.json").exists() and int(p.name.split("-")[1]) <= pend]
        if not cands:
            if not (self.run_dir() / f"state-{pend - pend % self.cfg['state_every']}.tmp").exists():
                lr.pop("best_state_pending", None)  # no state that early (the trainer's first one comes later)
            return
        src = max(cands, key=lambda p: int(p.name.split("-")[1]))
        dst = self.run_dir() / f"best_state-{src.name.split('-')[1]}"
        if not dst.exists():
            dst.mkdir()
            for f in src.iterdir():
                os.link(f, dst / f.name)
        for old in self.run_dir().glob("best_state-*"):
            if old != dst:
                shutil.rmtree(old, ignore_errors=True)
        lr.pop("best_state_pending", None)

    def state_near(self, step: int) -> Path | None:
        """Newest exact state at or before `step` (the trainer keeps two; anneal bases are kept apart)."""
        cands = [p for p in self.run_dir().glob("state-*") if (p / "STATE.json").exists() and int(p.name.split("-")[1]) <= step]
        for pat in ("anneal_base-*", "best_state-*"):
            cands += [p for p in self.run_dir().glob(pat) if (p / "STATE.json").exists() and int(p.name.split("-")[1]) <= step]
        cands.sort(key=lambda p: int(p.name.split("-")[1]))
        return cands[-1] if cands else None

    def restart_from_state(self, state: Path) -> None:
        """Stop the trainer and continue exactly from `state` (rollback / after an anneal), with the current LR scale."""
        s = self.seg()
        for pid in self.trainer_pids():
            sh(["kill", "-TERM", pid])
        for _ in range(60):
            if not self.trainer_pids():
                break
            time.sleep(5)
        for pid in self.trainer_pids():
            sh(["kill", "-9", pid])
        sh(["pm2", "stop", s["run"]], timeout=60)
        (self.run_dir() / "PAUSE").unlink(missing_ok=True)
        # the restored step's data and checkpoints come again: the abandoned branch's newer checkpoints are deleted, or
        # moved out of the way when something refers to them (best per competition, staged), since the new branch writes
        # step-N dirs of the same names
        n = int(state.name.split("-")[1])
        s["branch"] = s.get("branch", 0) + 1
        s["branch_from"] = n  # checkpoints up to n are the same weights in the new branch: never screened again
        for p in list(self.run_dir().glob("step-*")) + list(self.run_dir().glob("state-*")) + [self.run_dir() / "final"]:
            try:
                k = s["steps"] if p.name == "final" else int(p.name.split("-")[1])
            except ValueError:
                continue
            if not p.exists() or k <= n:
                continue
            if self.protected(p):
                self.relocate(p, self.run_dir() / f"b{s['branch'] - 1}-{p.name}")
            else:
                shutil.rmtree(p, ignore_errors=True)
        if state.name.startswith(("anneal_base", "best_state")):
            target = self.run_dir() / f"state-{n}"
            if not target.exists():
                target.mkdir()
                for f in state.iterdir():
                    os.link(f, target / f.name)
            state = target
        self.write_control(lr_scale=self.st["lr"]["scale"], anneal=0)
        self.launch(resume_state=state)
        for p in self.run_dir().glob("anneal_base-*"):  # state-N holds hard links to the same files
            if int(p.name.split("-")[1]) <= n:
                shutil.rmtree(p, ignore_errors=True)

    def relocate(self, src: Path, dst: Path) -> None:
        src.rename(dst)
        for b in self.st["best"].values():
            if b["path"] == str(src):
                b["path"], b["ckpt"] = str(dst), dst.name
        for r in self.st["ready"]:
            if r.get("path") == str(src):
                r["path"] = str(dst)
        for x in self.st["screens"]:
            if x.get("path") == str(src):
                x["path"] = str(dst)
        log(f"moved {src.name} of the abandoned branch to {dst.name} (referenced)")

    def start_anneal(self, step: int, why: str, k: int | None = None) -> None:
        lr = self.st["lr"]
        base = self.newest_state()
        if base is None:
            return
        keep = self.run_dir() / f"anneal_base-{base.name.split('-')[1]}"  # hard links: free, survives the trainer's cleanup
        if not keep.exists():
            keep.mkdir()
            for f in base.iterdir():
                os.link(f, keep / f.name)
        k = k or max(20, self.cfg["screen_every"])
        lr["anneal"], lr["anneal_from"], lr["anneal_base"] = k, step, str(keep)
        self.write_control(lr_scale=lr["scale"], anneal=k)
        self.event("anneal", steps=k, base=keep.name, why=why)

    def anneal_done(self, step: int) -> bool:
        lr = self.st["lr"]
        return bool(lr.get("anneal")) and step >= lr["anneal_from"] + lr["anneal"]

    # ---------------------------------------------------------------- screens
    def due_screens(self) -> list[Path]:
        br = self.seg().get("branch", 0)
        done = {(x["run"], x.get("branch", 0), x["ckpt"]) for x in self.st["screens"]}
        kept = {(x["run"], x["ckpt"]) for x in self.st["screens"] if x.get("step", 1 << 30) <= self.seg().get("branch_from", -1)}
        out = []
        for c in self.saved_ckpts():
            step = self.ckpt_step(c)
            if (self.seg()["run"], br, c.name) in done or (self.seg()["run"], c.name) in kept:
                continue
            if c.name == "final" or step % self.cfg["screen_every"] == 0 or self.anneal_done(step):
                out.append(c)
        return out

    def stop_trainer(self, why: str, wait_min: int = 90) -> bool:
        """CONTROL stop: the trainer writes a checkpoint + exact state after the current step and exits; True once it
        is gone (killed after wait_min). Used by screens (a paused trainer keeps ~156 GB/GPU with all 110B parameters
        trainable: the scorer OOMs next to it) and merge rounds; the caller relaunches from the newest state."""
        s = self.seg()
        if not self.trainer_pids():
            return True
        s["stopping"] = True
        self.write_control(stop=True)
        t0 = time.time()
        while self.trainer_pids() and time.time() - t0 < wait_min * 60:
            time.sleep(10)
        for pid in self.trainer_pids():
            sh(["kill", "-9", pid])
        s.pop("stopping", None)
        (self.run_dir() / "PAUSE").unlink(missing_ok=True)
        self.free_gpus(why)
        return True

    def pause(self) -> bool:
        """Pause the trainer for the scorer; False if it never acknowledged (it died: the GPUs are ours anyway)."""
        out = self.run_dir()
        if not self.trainer_pids():
            return True
        (out / "PAUSE").touch()
        t0 = time.time()
        while time.time() - t0 < 45 * 60:
            pd = out / "PAUSED"
            if pd.exists() and pd.stat().st_mtime >= (out / "PAUSE").stat().st_mtime - 1:
                return True
            if not self.trainer_pids():
                return True
            time.sleep(5)
        (out / "PAUSE").unlink(missing_ok=True)
        return False

    def resume(self) -> None:
        (self.run_dir() / "PAUSE").unlink(missing_ok=True)
        self.seg()["resumed_at"] = time.time()  # a 40-min screen must not look like a 40-min silent trainer

    def campaign(self, recipe: str, comps: list[str], samples: str, limit: int, tag: str, kings: dict) -> dict:
        """Score one checkpoint / recipe vs `kings` (kings.reigning); {comp: {sample: verdict}}."""
        kargs = sum((["--king", f"{c}={kings[c]['name']}:{kings[c]['bar']}"] for c in comps), [])
        cmd = [PY, "scripts/merge/campaign43.py", "--recipe", recipe, "--comps", ",".join(comps), *kargs, "--campaign", tag,
               "--limit", str(limit), "--sample", samples, "--replicas", "4", "--batch-size", "32",
               "--pingpong", "0" if self.trainer_pids() else "1"]
        env = {**os.environ, "TEUTONIC_EVAL_CACHE": str(D / "eval_cache"), "TEUTONIC_MODELS_DIRS": str(D / "models")}
        r = sh(cmd, timeout=4 * 3600, cwd=REPO, env=env)
        with (LOGS / f"autopilot_{self.comp}_screens.log").open("a") as h:
            h.write(f"\n### {time.strftime('%F %T')} {' '.join(cmd)}\n{r.stdout[-20000:]}\n{r.stderr[-6000:]}\n")
        saved = re.findall(r"^saved (\S+_duel\.json)$", r.stdout, re.M)
        if r.returncode != 0 or not saved:
            raise RuntimeError(f"campaign43 failed (rc {r.returncode}): {r.stderr[-300:]}")
        results = json.loads(Path(saved[-1]).read_text())["results"]
        specs = samples.split(",")
        out: dict = {}
        for i, res in enumerate(results):  # slots: for comp in comps for spec in specs (campaign43's order)
            c, spec = comps[i // len(specs)], specs[i % len(specs)]
            v = res["verdict"]
            out.setdefault(c, {})[spec] = {"mu": v["mu_hat"], "lcb": v["lcb"], "n": v.get("n"), "sample": res["sample"],
                                           "king": (res.get("king") or {}).get("label"),
                                           "sources": {k: round(x["mu_hat"], 6) for k, x in (v.get("source_scores") or {}).items()}}
        return out

    def screen(self, ck: Path) -> None:
        s = self.seg()
        step = self.ckpt_step(ck)
        self.st["phase"] = "screen"
        self.save()
        kings, missing = {}, []
        for c in COMPS:  # a king that is not ingested yet (a new one downloading) only drops its own competition
            try:
                kings.update(K.reigning([c]))
            except K.KingNotReady as exc:
                missing.append(f"{c}: {exc}")
        if self.comp not in kings:
            self.event("screen_skipped", ckpt=ck.name, why="; ".join(missing)[:300])
            self.st["screens"].append({"run": s["run"], "branch": s.get("branch", 0), "ckpt": ck.name, "step": step,
                                       "skipped": "; ".join(missing)[:300]})
            self.st["phase"] = "train"
            return self.save()
        strip_identity_files(ck)
        was_alive = bool(self.trainer_pids())
        self.stop_trainer(f"screen {ck.name}")  # exact state first; relaunched after the screen (main loop)
        self._stopped_for_screen = self._stopped_for_screen or was_alive
        paused = was_alive
        comps = [c for c in COMPS if c in kings]
        try:
            res = self.campaign(str(ck), comps, "tag:A", self.cfg["screen_limit"], f"ap_{self.comp}", kings)
        except Exception as exc:  # noqa: BLE001 - a failed screen must never stop training
            self.resume()
            self.event("screen_failed", ckpt=ck.name, why=str(exc)[:300])
            self.st["screens"].append({"run": s["run"], "branch": s.get("branch", 0), "ckpt": ck.name, "step": step, "failed": str(exc)[:300]})
            self.st["phase"] = "train"
            return self.save()
        self.extra_screens(comps, kings)
        row = {"run": s["run"], "branch": s.get("branch", 0), "ckpt": ck.name, "path": str(ck), "step": step, "t": time.time(),
               "kings": {c: kings[c]["name"] for c in comps}, "res": {c: res[c]["tag:A"] for c in res},
               "annealed": bool(self.st["lr"].get("anneal"))}
        self.st["screens"].append(row)
        summary = " ".join(f"{c.upper()} {row['res'][c]['mu']:+.6f}" for c in COMPS if c in row["res"])
        if missing:
            summary += f" (not screened: {', '.join(m.split(':')[0] for m in missing)})"
        self.event("screen", ckpt=f"{s['run']}/{ck.name}", res=summary, paused=paused)
        for c, v in row["res"].items():
            b = self.st["best"].get(c)
            if b is None or b.get("king_digest") != kings[c]["digest"] or v["mu"] > b["mu"]:  # a new king resets the best
                self.st["best"][c] = {"mu": v["mu"], "lcb": v["lcb"], "path": str(ck), "run": s["run"], "ckpt": ck.name,
                                      "king": kings[c]["name"], "king_digest": kings[c]["digest"], "t": time.time()}
        # confirmation: any competition close enough, on two independent full samples (still paused)
        for c in comps:
            v = row["res"].get(c)
            bar = kings[c]["bar"]
            if v is None or v["mu"] < bar + self.cfg["margin"] or v["lcb"] <= bar:
                continue
            if any(r.get("path") == str(ck) and r.get("comp") == c for r in self.st["ready"]):
                continue
            self.confirm(ck, c, kings[c], v)
        self.resume()
        self.st["phase"] = "train"
        self.save()

    def extra_screens(self, comps: list[str], kings: dict) -> None:
        """EXTRA_SCREENS = [{"path": <model dir>, "label": ..., "limit": 2400}]: score other models while the trainer is
        stopped for this screen anyway (no extra stop). A model that is not complete yet waits for the next screen."""
        req = self.dir / "EXTRA_SCREENS"
        todo = read_json(req, []) if req.exists() else []
        left = []
        for x in todo:
            mp = Path(x["path"])
            # write_merge.py writes the index after the last shard and the recipe next to the dir last of all
            done = (mp / "model.safetensors.index.json").exists() and ((mp / "SAVED").exists()
                                                                      or (mp.parent / f"{mp.name}.recipe.json").exists())
            if not done:
                left.append(x)
                continue
            try:
                res = self.campaign(str(mp), comps, "tag:A", int(x.get("limit", self.cfg["screen_limit"])), f"ap_{self.comp}_extra", kings)
                summary = " ".join(f"{c.upper()} {res[c]['tag:A']['mu']:+.6f}" for c in COMPS if c in res)
                self.st.setdefault("extra_screens", []).append({"t": time.time(), "path": str(mp), "label": x.get("label"),
                                                                 "res": {c: res[c]["tag:A"] for c in res}})
                self.event("extra_screen", label=x.get("label") or mp.name, res=summary)
            except Exception as exc:  # noqa: BLE001
                self.event("extra_screen_failed", label=x.get("label") or mp.name, why=str(exc)[:200])
        if todo:
            write_json(req, left) if left else req.unlink(missing_ok=True)

    def confirm(self, ck: Path, comp: str, king: dict, screen: dict) -> None:
        self.event("confirm", ckpt=str(ck), comp=comp, screen_mu=f"{screen['mu']:+.6f}", bar=king["bar"])
        try:
            res = self.campaign(str(ck), [comp], "tag:B,tag:C", self.cfg["confirm_limit"], f"ap_{self.comp}_confirm",
                                {comp: king})[comp]
        except Exception as exc:  # noqa: BLE001
            return self.event("confirm_failed", ckpt=str(ck), comp=comp, why=str(exc)[:300])
        ok = all(r["lcb"] > king["bar"] for r in res.values())
        entry = {"t": time.time(), "comp": comp, "path": str(ck), "king": king["name"], "king_digest": king["digest"],
                 "bar": king["bar"], "screen": screen, "full": res, "confirmed": ok}
        self.st.setdefault("confirms", []).append(entry)
        self.event("confirmed" if ok else "not_confirmed", comp=comp, ckpt=str(ck),
                   full=" ".join(f"{k} mu {r['mu']:+.6f} lcb {r['lcb']:+.6f}" for k, r in res.items()))
        if ok:
            self.stage(ck, comp, entry)

    def stage(self, ck: Path, comp: str, entry: dict) -> None:
        rec = stage_checkpoint(ck, comp, entry, label=f"{comp}_{ck.parent.name}_{ck.name}")
        self.st["ready"].append(rec)
        self.event("SUBMIT_READY", comp=comp, ckpt=str(ck), staged=rec["staged_dir"],
                   note="verify the stage log, then submit by hand: one submission per hotkey")

    # ---------------------------------------------------------------- end of a segment / anneal
    def after_screen(self, ck: Path) -> bool:
        """True when the trainer was (re)started here (cycle judged / anneal restored)."""
        lr = self.st["lr"]
        step = self.ckpt_step(ck)
        if lr.get("cycle") and lr.get("anneal") and self.anneal_done(step):
            self.judge_cycle(ck)
            return True
        if lr.get("anneal") and self.anneal_done(step):
            base = Path(lr["anneal_base"])
            lr["anneal"] = 0
            lr["ceiling"] = None  # a fresh climb after the restore
            # the restore continues at the same scale (it used to climb x1.3: raising on flat val only added noise)
            lr["since"] = int(base.name.split("-")[1])
            self.event("anneal_result", ckpt=ck.name, restore=base.name, next_scale=f"{lr['scale']:.3f}")
            self.restart_from_state(base)
            return True
        return False

    # ---------------------------------------------------------------- fixed test cycle
    def check_lr_request(self) -> None:
        """REQUEST_LR {"scale": x, "why": "..."}: set the LR scale live (CONTROL lr_scale) and keep it across relaunches."""
        req = self.dir / "REQUEST_LR"
        if not req.exists():
            return
        spec = read_json(req, {}) or {}
        req.unlink(missing_ok=True)
        if "scale" not in spec:
            return
        lr = self.st["lr"]
        step = max((r.get("step", 0) for r in self.read_log()), default=0)
        lr.setdefault("history", []).append({"step": step, "act": "set", "from": lr["scale"], "to": float(spec["scale"]),
                                             "why": spec.get("why", "requested")})
        lr["scale"] = float(spec["scale"])
        self.write_control(lr_scale=lr["scale"])
        self.event("lr", act="set", scale=f"{lr['scale']:.3f}", lr=f"{self.cfg['lr'] * lr['scale']:.2e}", why=spec.get("why", "requested"))

    def check_relaunch_request(self) -> None:
        """REQUEST_RELAUNCH: stop the trainer (checkpoint + exact state), relaunch the same run exactly from that state
        with the current configuration (e.g. pools that were still being built at the last launch)."""
        req = self.dir / "REQUEST_RELAUNCH"
        if not req.exists():
            return
        req.unlink(missing_ok=True)
        s = self.seg()
        if self.trainer_pids():
            s["stopping"] = True
            self.write_control(stop=True)
            t0 = time.time()
            while self.trainer_pids() and time.time() - t0 < 30 * 60:
                time.sleep(10)
            for pid in self.trainer_pids():
                sh(["kill", "-9", pid])
            s.pop("stopping", None)
        st = self.newest_state()
        self.event("relaunch", state=st and st.name, pools=",".join(n for n in self.cfg["pools"] if (POOLS / n / "meta.json").exists()))
        if st is not None:
            self.launch(resume_state=st)
        # no exact state at all (the trainer died before its first one): check_health's recovery handles the run dir

    def check_cycle_request(self) -> None:
        """REQUEST_CYCLE {"state": "best_state-40", "scale": 0.84, "hold": 60, "anneal": 40, "judge": "tag:C"}: continue
        exactly from that state, `hold` steps at `scale` with no LR decisions, then a cosine anneal of `anneal` steps; the
        end checkpoint and the state's own checkpoint are then scored on `judge` (a sample neither the merge pick nor the
        screens used) and the trainer stops for a decision (phase "hold"). One cycle answers "does training from here add
        anything" without the controller reacting to noise half-way."""
        req = self.dir / "REQUEST_CYCLE"
        if not req.exists():
            return
        spec = read_json(req, {}) or {}
        req.unlink(missing_ok=True)
        state = self.run_dir() / spec.get("state", "")
        ref = self.run_dir() / f"step-{state.name.split('-')[-1]}"
        if not (state / "STATE.json").exists() or not (ref / "SAVED").exists():
            return self.event("cycle_refused", why=f"needs an exact state and its checkpoint: {state.name}, {ref.name}")
        n = int(state.name.split("-")[1])
        lr = self.st["lr"]
        ref_val = [e["val"] for e in self._evals if e["step"] == n and e.get("val") is not None]
        lr.update(scale=float(spec.get("scale", lr["scale"])), since=n, ceiling=None, anneal=0,
                  best_val=ref_val[-1] if ref_val else None, best_step=n)
        lr["cycle"] = {"from": n, "ref": str(ref), "anneal_at": n + int(spec.get("hold", 60)), "k": int(spec.get("anneal", 40)),
                       "judge": spec.get("judge", "tag:C"), "scale": lr["scale"], "t": time.time()}
        self.event("cycle", state=state.name, scale=f"{lr['scale']:.3f}", hold=spec.get("hold", 60),
                   anneal=lr["cycle"]["k"], judge=lr["cycle"]["judge"])
        self.restart_from_state(state)

    def judge_cycle(self, ck: Path) -> None:
        lr, s = self.st["lr"], self.seg()
        cyc = lr.pop("cycle")
        lr["anneal"] = 0
        # the GPUs go to the judge, then wait for a decision: a CONTROL stop exits once its exact state is complete
        self.write_control(stop=True)
        for _ in range(240):
            if not self.trainer_pids():
                break
            time.sleep(5)
        for pid in self.trainer_pids():
            sh(["kill", "-TERM", pid])
        time.sleep(30)
        sh(["pm2", "stop", s["run"]], timeout=60)
        (self.run_dir() / "PAUSE").unlink(missing_ok=True)
        self.st["phase"] = "hold"
        self.save()
        out = {"ckpt": str(ck), "ref": cyc["ref"], "judge": cyc["judge"], **cyc}
        try:
            king = K.reigning([self.comp])
            for key, path in (("end", str(ck)), ("start", cyc["ref"])):
                r = self.campaign(path, [self.comp], cyc["judge"], self.cfg["confirm_limit"], f"ap_{self.comp}_cycle", king)
                out[key] = r[self.comp][cyc["judge"]]
            out["gain"] = round(out["end"]["mu"] - out["start"]["mu"], 6)
            self.event("cycle_result", comp=self.comp, end=f"{out['end']['mu']:+.6f}", start=f"{out['start']['mu']:+.6f}",
                       gain=f"{out['gain']:+.6f}", sample=cyc["judge"], note="trainer stopped (phase hold): decide next")
        except Exception as exc:  # noqa: BLE001
            out["error"] = str(exc)[:300]
            self.event("cycle_failed", why=str(exc)[:300])
        self.st.setdefault("cycles", []).append(out)
        self.save()

    def account_data(self, s: dict, steps_done: int) -> None:
        """Sequences of the lineage's data order this segment consumed (the next segment continues after them)."""
        if not s.get("accounted"):
            self.st["seqs_used"] = s.get("skip", 0) + steps_done * s.get("per_step", 2048)
            s["accounted"] = True

    def segment_done(self) -> None:
        """final saved and screened: continue from the best checkpoint of this segment for the target competition."""
        s = self.seg()
        if not any(x["run"] == s["run"] and x.get("branch", 0) == s.get("branch", 0) and x["ckpt"] == "final"
                   for x in self.st["screens"]):
            return  # the final screen runs first (main loop)
        mine = [x for x in self.st["screens"] if x["run"] == s["run"] and "res" in x and self.comp in x["res"]
                and Path(x.get("path", ""), "SAVED").exists()]
        best = max(mine, key=lambda x: x["res"][self.comp]["mu"], default=None)
        start = best["path"] if best else str(self.run_dir() / "final")
        self.account_data(s, s["steps"])
        self.event("segment_done", run=s["run"], next_start=start, best=best and f"{best['res'][self.comp]['mu']:+.6f}")
        self.new_segment(start, f"continue from the best of {s['run']}")

    # ---------------------------------------------------------------- kings / merge mode
    def check_kings(self, force: bool = False) -> None:
        if not force and time.time() - self._last_king_check < 300:
            return
        self._last_king_check = time.time()
        watch = [self.comp] + (["main"] if self.a.follow_main and self.comp != "main" else [])
        changed = []
        for c in COMPS:
            try:
                k = K.live(c)
            except K.KingNotReady:
                continue
            old = (self.st["kings"].get(c) or {}).get("digest")
            if old is None:
                self.st["kings"][c] = {"digest": k["digest"], "reign": k["reign"], "repo": k["repo"]}
            elif old != k["digest"]:
                if c in watch:
                    changed.append((c, k))
                else:
                    self.st["kings"][c] = {"digest": k["digest"], "reign": k["reign"], "repo": k["repo"]}
                    self.event("king_noted", comp=c, repo=k["repo"])
        req = self.dir / "REQUEST_MERGE"
        if not changed and not req.exists():
            return self.save()
        for c, k in changed:
            try:
                name, _ = K.resolve(k["digest"])
            except K.KingNotReady as exc:
                if not self.st.get("waiting_king") == k["digest"]:
                    self.st["waiting_king"] = k["digest"]
                    self.event("new_king_waiting", comp=c, repo=k["repo"], why=str(exc)[:160])
                return  # keep training until it is ingested
            self.event("new_king", comp=c, repo=k["repo"], name=name)
        req.unlink(missing_ok=True)
        for c, k in changed:
            self.st["kings"][c] = {"digest": k["digest"], "reign": k["reign"], "repo": k["repo"]}
        self.st.pop("waiting_king", None)
        self.save()
        self.merge_round("new king: " + ", ".join(f"{c} {k['repo']}" for c, k in changed) if changed else "requested")

    def merge_round(self, why: str) -> None:
        """Stop training, find the best start for training vs the current kings (merge4train.py), continue from it."""
        s = self.seg()
        self.st["phase"] = "merge"
        self.st["pending_merge"] = why
        self.event("merge_mode", why=why)
        if self.trainer_pids():
            s["stopping"] = True
            self.write_control(stop=True)  # checkpoint + exact state at the end of the current step
            t0 = time.time()
            while self.trainer_pids() and time.time() - t0 < 90 * 60:
                time.sleep(15)
            for pid in self.trainer_pids():
                sh(["kill", "-9", pid])
            s.pop("stopping", None)
        self.free_gpus("merge round")
        ours = self.ours()
        out = self.dir / f"merge_{time.strftime('%m%d%H%M', time.gmtime())}.json"
        cmd = [PY, "scripts/merge/merge4train.py", "--comp", self.comp, "--out", str(out)]
        for o in ours:
            cmd += ["--ours", f"{o['path']}::{o['start']}::{o['label']}"]
        r = sh(cmd, timeout=8 * 3600, cwd=REPO)
        with (LOGS / f"autopilot_{self.comp}_merge.log").open("a") as h:
            h.write(f"\n### {time.strftime('%F %T')} {' '.join(cmd)}\n{r.stdout[-30000:]}\n{r.stderr[-8000:]}\n")
        res = read_json(out, None)
        self.st.pop("pending_merge", None)
        if r.returncode != 0 or not res or not res.get("choice"):
            self.event("merge_failed", why=(r.stderr or r.stdout)[-300:])
            st = self.newest_state()
            return self.launch(resume_state=st) if st else self.new_segment(s["start"], "merge failed: restart")
        ch = res["choice"]
        for w in res.get("ready") or []:
            self.st["ready"].append(w)
            self.event("SUBMIT_READY", comp=w.get("comp"), staged=w.get("staged_dir"), note="merge winner; submit by hand")
        self.event("merge_choice", choice=ch["label"], kind=ch["kind"], mu=f"{ch['mu']:+.6f}", path=ch.get("path"))
        newest = self.saved_ckpts()[-1:] if self.seg() else []
        st = self.newest_state()
        if (ch["kind"] == "ours_as_is" and newest and ch.get("path") == str(newest[0]) and st is not None
                and int(st.name.split("-")[1]) == self.ckpt_step(newest[0])):
            return self.launch(resume_state=st)  # the current run is still the best start: continue exactly, nothing reset
        cks = self.saved_ckpts()
        self.account_data(s, self.ckpt_step(cks[-1]) if cks else 0)
        self.new_segment(ch["path"], f"merge round ({why}): {ch['label']}")

    def ours(self) -> list[dict]:
        """Our training results for merge4train: newest and best checkpoint of the current segment (its training delta
        = checkpoint - segment start), plus the best-ever checkpoint per competition from other segments."""
        out, seen = [], set()
        s = self.seg()
        if s:
            ck = self.saved_ckpts()
            if ck:
                out.append({"path": str(ck[-1]), "start": s["start"], "label": f"{s['run']}/{ck[-1].name}"})
            b = self.st["best"].get(self.comp)
            if b and Path(b["path"], "SAVED").exists():
                seg = next((g for g in self.st["segments"] if g["run"] == b["run"]), None)
                if seg:
                    out.append({"path": b["path"], "start": seg["start"], "label": f"{b['run']}/{b['ckpt']}"})
        res = []
        for o in out:
            if o["path"] not in seen and Path(o["path"], "model.safetensors.index.json").exists():
                seen.add(o["path"])
                res.append(o)
        return res

    # ---------------------------------------------------------------- retention
    def protected(self, p: Path) -> bool:
        ps = str(p)
        if any(ps == b["path"] for b in self.st["best"].values()):
            return True
        if any(ps == r.get("path") for r in self.st["ready"]):
            return True
        if any(ps == g["start"] for g in self.st["segments"]):
            return True
        return False

    def retain(self) -> None:
        if self.dry or not self.seg():
            return
        screened = {(x["run"], x.get("branch", 0), x["ckpt"]) for x in self.st["screens"]}
        for g in self.st["segments"]:
            rd = CKPT / g["run"]
            if not rd.exists():
                continue
            cks = self.saved_ckpts(rd)
            current = g is self.seg()
            keep_n = 2 if current else 0
            for c in cks[:-keep_n or None]:
                step_ok = (c.name == "final" or int(c.name.split("-")[1]) % self.cfg["screen_every"] != 0
                           or (g["run"], g.get("branch", 0), c.name) in screened)
                if self.protected(c) or not step_ok:
                    continue
                shutil.rmtree(c, ignore_errors=True)
                log(f"retention: deleted {g['run']}/{c.name}")
            if not current:  # finished segments: exact states are useless once the segment is over
                for p in list(rd.glob("state-*")) + list(rd.glob("anneal_base-*")):
                    shutil.rmtree(p, ignore_errors=True)
        for p in list(self.run_dir().glob("*.tmp")) + list(self.run_dir().glob("state-*.tmp")):
            if time.time() - p.stat().st_mtime > 3600:
                shutil.rmtree(p, ignore_errors=True)
        free = shutil.disk_usage(D).free / 1e12
        if free < 1.2:
            self.event("disk_low", free_tb=f"{free:.2f}")

    # ---------------------------------------------------------------- main loop
    def initial_start(self) -> tuple[str, str]:
        if self.a.start:
            p = Path(self.a.start)
            if (p / "model.safetensors.index.json").exists():
                return str(p.resolve() if p.is_symlink() else p), "given"
            models = read_json(REPO / "configs/merge/k41/models.json", {}) or {}
            if self.a.start in models:
                return models[self.a.start], f"model {self.a.start}"
            raise SystemExit(f"--start {self.a.start}: not a checkpoint dir or a models.json name")
        inits = read_json(SEARCH / "train_inits.json", {}) or {}
        ent = (inits.get("train_init") or inits).get(self.comp) if isinstance(inits, dict) else None
        if ent and ent.get("path") and Path(ent["path"], "model.safetensors.index.json").exists():
            return ent["path"], f"train_inits.json ({ent.get('label')})"
        k = K.reigning([self.comp])[self.comp]
        return k["path"], f"the reigning king {k['name']} (no merge or training result yet)"

    def run(self) -> None:
        if not self.st["segments"]:
            for c in COMPS:
                try:
                    k = K.live(c)
                    self.st["kings"][c] = {"digest": k["digest"], "reign": k["reign"], "repo": k["repo"]}
                except K.KingNotReady:
                    pass
            start, why = self.initial_start()
            others = [p for p in self.any_trainer() if p]
            if others and self.dry:
                log(f"DRY: another trainer is on the GPUs (pids {' '.join(others[:3])}); a real launch needs it stopped or --force")
            elif others and not self.a.force:
                raise SystemExit(f"another trainer is on the GPUs (pids {' '.join(others[:3])}); --force replaces it")
            if self.dry:
                self.st["segments"].append({"run": f"{self.comp}_ap_DRY", "start": start, "steps": self.cfg["steps"]})
                self.st["lr"] = {"scale": 1.0}
                log(f"DRY: start {start} ({why})")
                return self.launch()
            if others:
                for p in others:
                    sh(["kill", "-TERM", p])
                self.free_gpus("--force")
            self.new_segment(start, why)
        else:
            log(f"adopting lineage: segment {self.seg()['run']} ({len(self.st['segments'])} segments), "
                f"trainer {'alive' if self.trainer_pids() else 'not running'}")
            # a pause left by a screen that died with the previous autopilot process (pm2 stop kills its scorer)
            if (self.run_dir() / "PAUSE").exists() and not sh(["pgrep", "-f", "[s]cripts/merge/campaign43.py"]).stdout.strip():
                if not self.dry:
                    (self.run_dir() / "PAUSE").unlink(missing_ok=True)
                log("removed a stale PAUSE (no scorer is running)")
            self._seen_rows = 0
            self._evals = []
            rows = self.read_log()
            self._evals = [r for r in rows if "val" in r and r.get("step", 0) > 0]
            self._seen_rows = len(rows)
            if self.dry:
                lr = self.st["lr"]
                step = max((r.get("step", 0) for r in rows), default=0)
                log(f"DRY: LR decision now: {lr_decision(self._evals, lr, self.cfg, step)}")
                return
        if self.st.get("phase") == "needs_attention":
            while True:  # idle instead of a pm2 restart loop; `bash train.sh stop <comp>` and a fresh start reset it
                log(f"needs attention (see the last events in {self.state_path}); idle")
                time.sleep(3600)
        if self.st.get("pending_merge") and not self.dry:
            self.merge_round(f"resumed after a restart: {self.st['pending_merge']}")
        while True:
            try:
                if self.st.get("phase") == "hold":  # after a test cycle: the trainer waits for a decision
                    self.check_kings()
                    time.sleep(30)
                    continue
                self.check_cycle_request()
                self.check_relaunch_request()
                self.check_lr_request()
                self.check_health()
                if self.st.get("phase") == "needs_attention":
                    return
                self.control_lr()
                for ck in self.due_screens():
                    self.screen(ck)
                    restarted = self.after_screen(ck)
                    if self._stopped_for_screen and not restarted and not self.trainer_pids():
                        st_ = self.newest_state()  # the state the stop wrote: continue exactly
                        if st_ is not None:
                            self.launch(resume_state=st_)
                    self._stopped_for_screen = False
                if (self.run_dir() / "final" / "SAVED").exists() and not self.trainer_pids():
                    self.segment_done()
                self.check_kings()
                self.retain()
                self.save()
            except K.KingNotReady as exc:
                log(f"king not ready: {exc}")
            time.sleep(30)


def _q(x: str) -> str:
    import shlex
    return shlex.quote(str(x))


def _tail(path: Path, n: int) -> str:
    try:
        with open(path, "rb") as h:
            h.seek(0, 2)
            size = h.tell()
            h.seek(max(0, size - n))
            return h.read().decode(errors="replace")
    except OSError:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comp", required=True, choices=COMPS)
    ap.add_argument("--start", default=None, help="first segment's start: checkpoint dir or models.json name (default: "
                    "search/train_inits.json for the competition, else the reigning king)")
    ap.add_argument("--follow-main", type=int, default=1, help="a new MAIN king also triggers a merge round (specialists)")
    ap.add_argument("--force", action="store_true", help="replace a trainer that is not ours")
    ap.add_argument("--dry", action="store_true")
    for k in ("lr", "wd", "dense_frac", "scale_max", "rollback_rise", "decay_frac"):
        ap.add_argument(f"--{k.replace('_', '-')}", type=float, default=None)
    ap.add_argument("--policy", choices=("wsd", "ema"), default=None, help="LR policy: wsd (default: scheduled hold + decay, "
                    "screens judge) or ema (adaptive raise/lower/anneal on the EMA-val trend)")
    for k in ("warmup", "steps", "screen_every", "save_every", "state_every", "eval_every", "screen_limit", "val_n"):
        ap.add_argument(f"--{k.replace('_', '-')}", type=int, default=None)
    ap.add_argument("--ema", type=float, default=None)
    ap.add_argument("--freeze", default=None)
    ap.add_argument("--batches", default=None, help="per-GPU micro-batch x accumulation, e.g. 16x16")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # pm2 stop: the trainer keeps running; a restart adopts it
    Pilot(args).run()


if __name__ == "__main__":
    main()
