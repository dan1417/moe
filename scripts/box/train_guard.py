#!/usr/bin/env python3
"""Watchdog for the MAIN training lineage (user 15:05: "adjust the training pipeline so the training goes well and stably").

Every 60 s: is the current run finished, crashed (trainer process gone without a final checkpoint) or hung (trainer log
silent for 25 min while not paused)? On a crash or hang: stop the run's jobs, free the GPUs, and resume from its newest
complete checkpoint into a new run <base>_r<k> that continues the lineage exactly -- data order (--skip-seqs), remaining
steps, and the warmup-stable-decay LR schedule (also when the decay had already started) -- then restart the step
watcher and the checkpoint retention for the new run. At most 3 resumes. State: /data/teutonic/ckpt/<base>.lineage.json.

  train_guard.py [--dry]      (--dry: decide and print the resume command once, change nothing)
"""
import argparse
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path

D = Path("/data/teutonic")
CKPT, LOGS = D / "ckpt", D / "logs"
# the lineage's plan: /data/teutonic/ckpt/current_plan.json (written by switch_training.sh when training restarts from a
# better start); default = the m3 plan
_plan_file = CKPT / "current_plan.json"
PLAN = {"base": "main_upper_ema_m3", "start": str(CKPT / "main_upper_ema_m1/step-175"), "total": 425, "skip0": 358400,
        "decay_abs": 340, "lr": 1.5e-6, "min_frac": 0.1, "warmup": 10, "seqs_per_step": 2048, "max_resumes": 3}
if _plan_file.exists():
    PLAN.update(json.loads(_plan_file.read_text()))
BASE = PLAN["base"]
# a hang = no log line for 3x the step-log interval (10 steps at ~105 s per 2 048 seqs), at least 25 min: at 4 096 seqs/step the
# trainer logs every ~35 min (02:26 JST Oct 12: a 25 min threshold restarted a healthy c7 as a "hang")
HANG_S = max(25 * 60, int(3 * 10 * 105 * PLAN.get("seqs_per_step", 2048) * PLAN.get("seq_len", 2048) / 2048 ** 2))  # tokens per step


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S', time.gmtime())} {msg}", flush=True)


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, text=True, capture_output=True).stdout


def pm2_status() -> dict:
    try:
        return {p["name"]: p["pm2_env"]["status"] for p in json.loads(sh("pm2 jlist"))}
    except json.JSONDecodeError:
        return {}


def lr_at(step: int) -> float:
    """The lineage's LR at absolute step `step` (warmup ignored: a resume re-warms over 5 steps)."""
    if step < PLAN["decay_abs"]:
        return PLAN["lr"]
    prog = (step - PLAN["decay_abs"]) / max(1, PLAN["total"] - PLAN["decay_abs"])
    return PLAN["lr"] * (PLAN["min_frac"] + (1 - PLAN["min_frac"]) * 0.5 * (1 + math.cos(math.pi * prog)))


def saved_steps(run_dir: Path) -> list[int]:
    out = []
    for p in run_dir.glob("step-*"):
        m = re.fullmatch(r"step-(\d+)", p.name)
        if m and not p.is_symlink() and (p / "SAVED").exists():
            out.append(int(m.group(1)))
    return sorted(out)


def state_path() -> Path:
    return CKPT / f"{BASE}.lineage.json"


def load_state() -> dict:
    p = state_path()
    if p.exists():
        return json.loads(p.read_text())
    return {"runs": [{"run": BASE, "offset": 0, "start": PLAN["start"]}], "resumes": 0, "events": []}  # fresh lineage


def stop_jobs(run: str) -> None:
    for name in ("main_watch", "ckpt_retention", f"launch_{run}", "main_train3", f"launch_{BASE}", run):
        sh(f"pm2 stop {name} >/dev/null 2>&1; pm2 delete {name} >/dev/null 2>&1")
    (CKPT / run / "PAUSE").unlink(missing_ok=True)
    for _ in range(36):  # wait for the GPUs to empty; kill leftovers of the dead run
        pids = [x for x in sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader").split() if x.strip()]
        if not pids:
            return
        if _ > 6:
            sh("kill -9 " + " ".join(pids))
        time.sleep(5)


def resume_plan(state: dict) -> dict:
    cur = state["runs"][-1]
    run_dir = CKPT / cur["run"]
    steps = saved_steps(run_dir)
    if steps:
        done, start = cur["offset"] + steps[-1], str(run_dir / f"step-{steps[-1]}")
    else:  # nothing saved in this run: restart it from its own start
        done, start = cur["offset"], cur["start"]
    k = state["resumes"] + 1
    lr_now = lr_at(done)
    remaining = PLAN["total"] - done
    if done >= PLAN["decay_abs"]:  # already decaying: cosine from the current LR down to the same final LR
        lr, decay_start, min_frac = lr_now, 0, PLAN["lr"] * PLAN["min_frac"] / lr_now
    else:
        lr, decay_start, min_frac = PLAN["lr"], PLAN["decay_abs"] - done, PLAN["min_frac"]
    return {"run": f"{BASE}_r{k}", "start": start, "offset": done, "steps": remaining, "skip": PLAN["skip0"] + done * PLAN["seqs_per_step"],
            "warmup": 5, "decay_start": decay_start, "lr": lr, "min_frac": min_frac}


def launch(p: dict) -> None:
    env = (f"M_RUN={p['run']} M_START={p['start']} M_STEPS={p['steps']} M_SKIP={p['skip']} M_WARMUP={p['warmup']} "
           f"M_DECAY_START={p['decay_start']} M_LR={p['lr']:.4e} M_MIN_LR_FRAC={p['min_frac']:.4f} "
           f"M_DENSE_FRAC={PLAN.get('dense_frac', 2.0)} M_VAL_RAW={PLAN.get('val_raw', 0)}"
           + (f' M_FREEZE="{PLAN["freeze"]}"' if PLAN.get("freeze") else "")
           + (f" M_POOLS={PLAN['pools']} M_VAL={PLAN['val']}" if PLAN.get("pools") else "")
           + f" M_BATCHES={PLAN.get('batches', '16x16')} M_SAVE_EVERY={PLAN.get('save_every', 25)} M_EMA={PLAN.get('ema', 0.98)}"
           + f" M_SAVE_SR={PLAN.get('save_sr', 0)} M_EVAL_EVERY={PLAN.get('eval_every', PLAN.get('save_every', 25))}")
    sh(f"pm2 start bash --name launch_{p['run']} --no-autorestart --time --merge-logs --output {LOGS}/pm2_launch_{p['run']}.log "
       f"--error {LOGS}/pm2_launch_{p['run']}.log -- -c '{env} exec bash {LOGS}/{PLAN.get('launcher', 'main_run.sh')}' >/dev/null")
    sh(f"pm2 start bash --name main_watch --no-autorestart --output {LOGS}/pm2_main_watch.log --error {LOGS}/pm2_main_watch.log "
       f"-- -c 'COMPS={PLAN.get('comps', 'main,math')} EVERY={PLAN.get('screen_every', PLAN.get('save_every', 25))} SR={1 if PLAN.get('save_sr') == 2 else 0} RUN={CKPT / p['run']} "
       f"exec bash {LOGS}/main_watch.sh {PLAN.get('screen_every', PLAN.get('save_every', 25))}' >/dev/null")
    sh(f"pm2 start bash --name ckpt_retention --no-autorestart --output {LOGS}/pm2_ckpt_retention.log --error {LOGS}/pm2_ckpt_retention.log "
       f"-- -c 'RUN={CKPT / p['run']} KEEP=none exec bash {LOGS}/ckpt_retention.sh' >/dev/null")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    log(f"guard for {BASE}: crash = trainer gone without final, hang = log silent {HANG_S // 60} min while not paused")
    while True:
        state = load_state()
        cur = state["runs"][-1]
        run, run_dir = cur["run"], CKPT / cur["run"]
        status = pm2_status().get(run)
        if (run_dir / "final" / "SAVED").exists():
            log(f"{run}: finished (final saved); guard exits")
            return
        problem = None
        if not run_dir.exists():
            problem = None  # not started yet
        elif status not in ("online", "launching"):
            if status is not None or saved_steps(run_dir) or (run_dir / "train_log.jsonl").exists():
                problem = f"crash (trainer pm2 status {status})"
        else:
            out_log = LOGS / f"pm2_{run}.out.log"
            silent = time.time() - out_log.stat().st_mtime if out_log.exists() else 0
            if silent > HANG_S and not (run_dir / "PAUSE").exists():
                problem = f"hang (trainer log silent {silent / 60:.0f} min, not paused)"
        if problem or args.dry:
            plan = resume_plan(state)
            log(f"{run}: {problem or 'DRY RUN'} -> resume plan {json.dumps(plan)}")
            if plan["steps"] <= 0:
                log(f"{run}: all {PLAN['total']} lineage steps are done (no final dir); nothing to resume, guard exits")
                return
            if args.dry:
                return
            if state["resumes"] >= PLAN["max_resumes"]:
                log(f"GIVING UP: {state['resumes']} resumes already; needs a look")
                return
            stop_jobs(run)
            launch(plan)
            state["resumes"] += 1
            state["runs"].append({"run": plan["run"], "offset": plan["offset"], "start": plan["start"]})
            state["events"].append({"t": time.time(), "problem": problem, "plan": plan})
            state_path().write_text(json.dumps(state, indent=1))
            log(f"resumed as {plan['run']} ({plan['steps']} steps left from absolute step {plan['offset']})")
        time.sleep(60)


if __name__ == "__main__":
    main()
