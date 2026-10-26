#!/usr/bin/env python3
"""Render a live HTML dashboard for a cpt_fsdp.py run, plus <out dir>/data.json that the open page polls every 10 s.

  dashboard.py --run /data/teutonic/ckpt/text_lora_a --out /data/teutonic/www/index.html [--name "TEXT LoRA a"] [--target 0.0026]

Everything shown is read from the box, nothing is typed in by hand:
  * the recipe (mode, lr, schedule, batch, pool, val set) from the launcher script /data/teutonic/logs/run_<run>.sh
    (the exact cpt_fsdp.py command line; later flags win, as in argparse);
  * the phase (loading / FSDP init / step-0 val / training / validating / saving / paused for scoring / finished /
    crashed) and the wall-clock time of every event from the pm2 log /data/teutonic/logs/pm2_<run>.out.log and the
    per-step rows of <run>/train_log.jsonl (steps, val, timestamps);
  * per-step throughput (pure training time, without val/save/scoring pauses) and an ETA that adds the measured
    val/save/scoring durations for the checkpoints still ahead;
  * live GPU utilization/memory/power (nvidia-smi), every run under /data/teutonic/ckpt with a scores.json (today's
    experiments side by side), and background jobs (king downloads, merge writes);
  * the autopilot panel (scripts/train/autopilot.py): <TEUTONIC_AUTOPILOT_DIR, default /data/teutonic/autopilot>/<comp>/
    status.json (+ state.json for the chart markers: LR-scale changes, rollbacks, anneals) and the staged submission
    candidates in search/autopilot_ready.json (TEUTONIC_AUTOPILOT_READY); on top of the page while a status is < 1 h old.
All times are JST (GMT+9); the autopilot panel shows UTC and JST.

scores.json (written by score_ckpts.py): [{"ckpt": "step-100", "text_mu": 0.0012, "text_se": 0.0004, "lcb": ...,
  "splits": {"text": 0.001, "math": 0.0, "code": 0.0}, "n": 2400, "status": "done|scoring|pending", "note": ""}, ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shlex
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path

LOGS = Path("/data/teutonic/logs")
CKPT = Path("/data/teutonic/ckpt")
MODELS = Path("/data/teutonic/models")
JST = dt.timezone(dt.timedelta(hours=9))
PM2_LINE = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d): (.*)$")
STEP_LINE = re.compile(r"^step (\d+)/(\d+): train ([\d.]+) \|g\| ([\d.]+) lr ([\d.e+-]+) ([\d.]+)k tok/s")
VAL_LINE = re.compile(r"^step (\d+): val ([\d.]+)")

TEMPLATE = r"""<meta charset="utf-8">
<title>__NAME__</title>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>
/* layout: a control-room strip (status + now + tiles) over a two-column chart grid; tables below */
:root {
  color-scheme: light;
  --bg: #f3f5f8;          /* cool off-white, slight blue bias */
  --surface: #ffffff;
  --ink: #141a22;
  --ink-2: #4b5563;
  --ink-3: #8a94a3;
  --line: #d9dee6;
  --grid: #e6eaf0;
  --accent: #1f5fb8;      /* val CE */
  --train: #d9602c;       /* train loss */
  --aux: #15957f;         /* |g| / lr / throughput */
  --good: #1a7f37; --warn: #b7791f; --crit: #c0392b;
  --good-bg: #e6f4ea; --warn-bg: #fdf3e1; --crit-bg: #fbe9e7;
  --font-display: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-body: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --font-mono: ui-monospace, "SFMono-Regular", Menlo, Consolas, "Liberation Mono", monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --bg: #0f1419; --surface: #171d24; --ink: #e8edf3; --ink-2: #aab4c0; --ink-3: #6f7b89;
    --line: #2a333d; --grid: #222a33; --accent: #5b9cf0; --train: #f08a5a; --aux: #3fc2a8;
    --good: #4cc26a; --warn: #e2ad4a; --crit: #ef6b5e; --good-bg: #16301f; --warn-bg: #33290f; --crit-bg: #3a1a16;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --bg: #0f1419; --surface: #171d24; --ink: #e8edf3; --ink-2: #aab4c0; --ink-3: #6f7b89;
  --line: #2a333d; --grid: #222a33; --accent: #5b9cf0; --train: #f08a5a; --aux: #3fc2a8;
  --good: #4cc26a; --warn: #e2ad4a; --crit: #ef6b5e; --good-bg: #16301f; --warn-bg: #33290f; --crit-bg: #3a1a16;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--font-body); font-size: 14px; line-height: 1.45;
  padding-inline: 20px; padding-block: 20px 40px; }
.wrap { max-width: 1880px; margin: 0 auto; display: grid; grid-template-columns: minmax(0, 1fr); gap: 18px; }  /* wide tables scroll inside their panel, not the page */
header { display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px 18px; }
h1 { font-family: var(--font-display); font-weight: 700; font-stretch: 85%; font-size: 26px; letter-spacing: -0.01em; margin: 0; text-wrap: balance; }
h2 { font-family: var(--font-display); font-size: 14px; font-weight: 600; margin: 0 0 8px; }
.sub { color: var(--ink-2); font-size: 13px; }
.live { font-size: 12px; color: var(--ink-3); font-variant-numeric: tabular-nums; }
.live.stale { color: var(--crit); font-weight: 600; }
.pill { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
.pill::before { content: ""; width: 8px; height: 8px; border-radius: 50%; background: currentColor; }
.pill.running, .pill.merging { background: var(--good-bg); color: var(--good); }
.pill.queued { background: var(--warn-bg); color: var(--warn); }
.pill.stopped, .pill.crashed { background: var(--crit-bg); color: var(--crit); }
.pill.finished, .pill.paused, .pill.busy { background: var(--warn-bg); color: var(--warn); }
.pill.idle { background: var(--grid); color: var(--ink-2); }
.now { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; display: grid; gap: 10px; }
.now .phase { font-size: 16px; font-weight: 600; }
.bar { height: 8px; background: var(--grid); border-radius: 999px; overflow: hidden; }
.bar > div { height: 100%; background: var(--accent); border-radius: 999px; }
.scorehead { display: flex; justify-content: space-between; gap: 12px; font-size: 13px; font-weight: 650; }
.scorehead b { font-variant-numeric: tabular-nums; }
.bar.score { height: 18px; }
.bar.score > div { background: linear-gradient(90deg, #1f6fe0, #12a88a); }
.events { display: flex; flex-wrap: wrap; gap: 6px 22px; font-size: 13px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.events b { color: var(--ink); font-weight: 600; }
.hero { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.compcard { border-radius: 16px; padding: 16px 18px 14px; color: #fff; min-height: 168px; display: grid; align-content: start; gap: 4px;
  box-shadow: 0 10px 28px rgba(0,0,0,.18); position: relative; overflow: hidden; }
.compcard::after { content: ""; position: absolute; right: -20px; top: -30px; width: 120px; height: 120px; border-radius: 50%; background: rgba(255,255,255,.12); }
.compcard.main { background: linear-gradient(145deg, #0d2f66, #1f6fe0); }
.compcard.math { background: linear-gradient(145deg, #084840, #12a88a); }
.compcard.code { background: linear-gradient(145deg, #6a2e0c, #e86a2a); }
.compcard.text { background: linear-gradient(145deg, #4a3208, #e0a020); }
.compcard.val { background: linear-gradient(145deg, #232a36, #5a6679); }
.compcard.up { outline: 3px solid #3DDC84; outline-offset: 0; }
.lg { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin: 0 4px 0 8px; vertical-align: baseline; }
.compcard .k { font-size: 12px; letter-spacing: .14em; text-transform: uppercase; opacity: .85; font-weight: 700; }
.compcard .mu { font-family: var(--font-mono); font-size: 36px; font-weight: 750; letter-spacing: -0.03em; line-height: 1.05; }
.compcard .who { font-size: 13px; opacity: .92; }
.compcard .how { font-size: 12px; opacity: .8; min-height: 2.4em; }
.gradrow { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.gchip { font-family: var(--font-mono); font-size: 11px; padding: 3px 8px; border-radius: 999px; background: var(--grid); color: var(--ink-2); }
.gchip.on { background: var(--good-bg); color: var(--good); font-weight: 700; }
.gchip.hot { background: #123a22; color: #b6f5c8; }
@media (max-width: 900px) { .hero { grid-template-columns: 1fr 1fr; } }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; }
.tile { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; display: grid; gap: 4px; }
.tile .k { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink-3); }
.tile .v { font-family: var(--font-mono); font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .d { font-size: 12px; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.tile .d.good { color: var(--good); } .tile .d.crit { color: var(--crit); }
.charts { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(640px, 100%), 1fr)); gap: 18px; }
.chart { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 16px 18px 12px; min-width: 0; }
.chart h2 { margin: 0 0 4px; font-size: 18px; }
.chart .note { font-size: 13px; }
.chart .lg { width: 12px; height: 12px; }
.chart .note, .panel .note { font-size: 12px; color: var(--ink-3); margin: 0 0 6px; }
.chart svg { width: 100%; height: auto; display: block; overflow: visible; }
.chart text { font-family: var(--font-mono); font-size: 13px; fill: var(--ink-2); font-variant-numeric: tabular-nums; }
.chart .grid line { stroke: var(--grid); stroke-width: 1; }
.chart .axis line { stroke: var(--line); }
.chart .lbl { fill: var(--ink-3); }
.tip { position: fixed; pointer-events: none; background: var(--surface); border: 1px solid var(--line); border-radius: 6px; padding: 7px 10px;
  font-family: var(--font-mono); font-size: 13px; color: var(--ink); box-shadow: 0 4px 14px rgba(0,0,0,0.12); display: none; z-index: 10; white-space: nowrap; }
.panel { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; overflow-x: auto; }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 14px; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--ink-3); font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
td.l { text-align: left; color: var(--ink-2); white-space: normal; min-width: 220px; }
td.num { font-family: var(--font-mono); font-size: 12.5px; }
td.pos { color: var(--good); } td.neg { color: var(--crit); }
tr.cur td { background: var(--good-bg); }
td .st { display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 11px; font-weight: 600; }
.st.done { background: var(--good-bg); color: var(--good); } .st.scoring, .st.busy { background: var(--warn-bg); color: var(--warn); }
.st.train { background: var(--surface, #fff); color: var(--good); border: 1px solid var(--good); }
.st.pending, .st.idle { background: var(--grid); color: var(--ink-2); } .st.below, .st.crit { background: var(--crit-bg); color: var(--crit); }
.meter { display: inline-block; width: 70px; height: 6px; background: var(--grid); border-radius: 999px; vertical-align: middle; margin-right: 6px; overflow: hidden; }
.meter > i { display: block; height: 100%; background: var(--aux); }
.foot { font-size: 12px; color: var(--ink-3); display: flex; flex-wrap: wrap; gap: 6px 18px; }
.foot span { min-width: 0; overflow-wrap: anywhere; }  /* the 16-pool list is one long word */
.empty { color: var(--ink-3); font-size: 13px; padding: 18px 0; }
td.dim { opacity: .55; } .vline { padding: 4px 0; font-size: 13px; }
/* autopilot: one card per competition controller, staged submissions on top */
.apgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(440px, 100%), 1fr)); gap: 14px; }
.apcard { border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; display: grid; gap: 8px; align-content: start; min-width: 0; overflow-x: auto; }
.apcard.old { opacity: .7; }
.apcard .hd { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 12px; }
.apcard .hd .c { font-family: var(--font-display); font-size: 17px; font-weight: 750; letter-spacing: .06em; }
.apcard .kv { font-size: 12.5px; color: var(--ink-2); display: grid; gap: 3px; }
.apcard .kv b { color: var(--ink); font-family: var(--font-mono); font-weight: 600; }
.apcard .kv .mono, .ready .mono { font-family: var(--font-mono); font-size: 12px; overflow-wrap: anywhere; }
.apcard h3 { font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--ink-3); margin: 4px 0 0; font-weight: 600; }
.apcard th, .apcard td { padding: 3px 7px; }
.apcard tr.own td:first-child { font-weight: 700; }
.pill.ap-train { background: var(--good-bg); color: var(--good); }
.pill.ap-screen { background: color-mix(in srgb, var(--accent) 16%, transparent); color: var(--accent); }
.pill.ap-merge { background: var(--warn-bg); color: var(--warn); }
.pill.ap-needs_attention { background: var(--crit-bg); color: var(--crit); }
.pill.ap-starting { background: var(--grid); color: var(--ink-2); }
.readybox { display: grid; gap: 10px; margin-bottom: 12px; }
.ready { border: 2px solid var(--good); background: var(--good-bg); border-radius: 8px; padding: 10px 14px; display: grid; gap: 4px; font-size: 13px; }
.ready.former { border-color: var(--line); background: var(--surface); opacity: .75; }
.ready .big { font-family: var(--font-display); font-size: 16px; font-weight: 750; color: var(--good); letter-spacing: .02em; }
.ready.former .big { color: var(--ink-2); }
.evlist { font-size: 12px; display: grid; gap: 3px; font-variant-numeric: tabular-nums; color: var(--ink-2); }
.evlist .t { color: var(--ink-3); font-family: var(--font-mono); font-size: 11.5px; margin-right: 6px; }
.evlist b { color: var(--ink); }
.evlist .hot b { color: var(--good); } .evlist .bad b { color: var(--crit); }
@media (prefers-reduced-motion: no-preference) { .pill.ap-train::before, .pill.ap-screen::before { animation: blink 1.6s ease-in-out infinite; } }
@media (max-width: 480px) { body { padding-inline: 16px; } .tile .v { font-size: 19px; } }
@media (prefers-reduced-motion: no-preference) { .pill.running::before, .pill.merging::before { animation: blink 1.6s ease-in-out infinite; } }
@keyframes blink { 50% { opacity: 0.35; } }
</style>
<div class="wrap">
  <header>
    <h1>__NAME__</h1>
    <span id="status" class="pill idle">idle</span>
    <span class="live" id="live"></span>
  </header>
  <div class="sub" id="subline"></div>
  <section class="panel" id="sec-ap" style="display:none">
    <h2>Autopilot</h2>
    <p class="note" id="n-ap"></p>
    <div class="readybox" id="ap-ready"></div>
    <div class="apgrid" id="ap"></div>
  </section>
  <section class="hero" id="hero"></section>
  <section class="now" id="now"></section>
  <section class="tiles" id="tiles"></section>
  <section class="charts">
    <div class="chart"><h2>Held-out CE (val)</h2><p class="note" id="n-val"></p><div id="c-val"></div></div>
    <div class="chart" id="w-split"><h2>Val CE by source (change since step 0)</h2><p class="note" id="n-split"></p><div id="c-split"></div></div>
    <div class="chart"><h2>Train loss</h2><p class="note" id="n-train"></p><div id="c-train"></div></div>
    <div class="chart"><h2>Throughput</h2><p class="note">tokens/s of each logged step, training time only (val, saves and scoring pauses excluded)</p><div id="c-tps"></div></div>
    <div class="chart"><h2>Gradient norm</h2><p class="note" id="n-g"></p><div id="c-g"></div></div>
    <div class="chart"><h2>Learning rate</h2><p class="note" id="n-lr"></p><div id="c-lr"></div></div>
  </section>
  <section class="panel">
    <h2 id="ledger-title">Checkpoints of this run on the TEXT screen vs T1</h2>
    <p class="note" id="ledger-note">paired mean gain (king CE − checkpoint CE) on the first 2400 sequences of the text-competition sample; bar 0.002, safe target +__TARGET__; Δ = vs this run's step 0</p>
    <div id="ledger"></div>
  </section>
  <section class="panel">
    <h2>Evaluation: full validator-size duels</h2>
    <p class="note">validator scoring since 14a9bb2 (document-masked, token-weighted) on full samples of 30 000 windows + ~1 670 whole 2K–8K documents; one row per checkpoint and king; ✅ = LCB above the bar (TEXT 0.002, MAIN 0.003); mean over the full samples; screens (first 2 400) in grey when no full result exists; hover a cell for LCB and n</p>
    <div id="evals"></div>
  </section>
  <section class="panel" id="sec-camp" style="display:none">
    <h2>Merge campaign: every recipe on all four competitions</h2>
    <p class="note" id="n-camp"></p>
    <div id="camp"></div>
  </section>
  <section class="two">
    <div class="panel"><h2>Eval jobs</h2><p class="note">pm2 eval_* / sample_* jobs (training is paused while a duel scores and resumes by itself)</p><div id="evaljobs"></div></div>
    <div class="panel"><h2>Validator duel (our submission)</h2><p class="note">live TEXT competition dashboard, our uid</p><div id="vduel"></div></div>
  </section>
  <section class="panel">
    <h2>All runs on this box</h2>
    <p class="note">every run directory with a screen score; Δ start = best scored checkpoint minus the run's own step 0 (paired, same 2400 sequences); val Δ = latest held-out CE minus step 0 (lower is better)</p>
    <div id="runs"></div>
  </section>
  <section class="two">
    <div class="panel"><h2>GPUs</h2><p class="note">nvidia-smi at render time</p><div id="gpus"></div></div>
    <div class="panel"><h2>Background jobs</h2><p class="note">king downloads and merge writes (CPU / network; the GPUs stay with training)</p><div id="jobs"></div></div>
  </section>
  <div class="foot" id="foot"></div>
</div>
<div class="tip" id="tip"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
  const fmt = (x, d=5) => (x==null || isNaN(x)) ? '–' : Number(x).toFixed(d);
  const sfmt = (x, d=5) => (x==null || isNaN(x)) ? '–' : (x>=0?'+':'') + Number(x).toFixed(d);
  const sgn = x => (x==null || isNaN(x)) ? '' : (x > 0 ? 'pos' : x < 0 ? 'neg' : '');
  const hms = s => { if (s==null||isNaN(s)) return '–'; s=Math.max(0,Math.round(s)); const h=Math.floor(s/3600), m=Math.floor(s%3600/60), x=s%60;
    return h ? `${h}h ${String(m).padStart(2,'0')}m` : (m ? `${m}m ${String(x).padStart(2,'0')}s` : `${x}s`); };
  const jst = e => e==null ? '–' : new Date(e*1000).toLocaleTimeString('en-GB', {timeZone:'Asia/Tokyo', hour:'2-digit', minute:'2-digit', hour12:false});
  const jsts = e => e==null ? '–' : new Date(e*1000).toLocaleTimeString('en-GB', {timeZone:'Asia/Tokyo', hour12:false});
  const esc = s => String(s==null?'':s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const utc = e => e==null ? '–' : new Date(e*1000).toLocaleTimeString('en-GB', {timeZone:'UTC', hour:'2-digit', minute:'2-digit', hour12:false});
  const when = e => { if (e==null || isNaN(e)) return '–';  // UTC + JST, with the JST date once it is older than 20 h
    const day = Date.now()/1000 - e > 20*3600 ? new Date(e*1000).toLocaleDateString('en-GB', {timeZone:'Asia/Tokyo', month:'2-digit', day:'2-digit'}) + ' ' : '';
    return `${utc(e)} UTC · ${day}${jst(e)} JST`; };
  const cs = getComputedStyle(document.documentElement);
  const C = { accent: cs.getPropertyValue('--accent').trim() || '#1f5fb8', train: cs.getPropertyValue('--train').trim() || '#d9602c', aux: cs.getPropertyValue('--aux').trim() || '#15957f' };
  const tip = document.getElementById('tip');
  let D = JSON.parse(document.getElementById('data').textContent);
  let sig = '';

  function render() {
    const vals = D.val, tr = D.train, R = D.now || {};
    const last = tr.length ? tr[tr.length-1] : null;
    const v0 = vals.length ? vals[0] : null, vl = vals.length ? vals[vals.length-1] : null;
    const vbest = vals.length ? vals.reduce((a,b)=> b.val < a.val ? b : a) : null;
    const st = document.getElementById('status');
    st.className = 'pill ' + D.status; st.textContent = D.status;
    const CP0 = D.campaign || {};
    if (D.status === 'merging') {
      document.getElementById('subline').textContent = 'Public-model merge search · the eight B300s are scoring recipes, training is paused';
      const P = CP0.progress || {};
      const pct = P.total ? Math.max(0, Math.min(100, 100 * P.done / P.total)) : 0;
      const qty = !P.total ? '' : (P.unit === 'bytes'
        ? `${(P.done / 1e9).toFixed(1)} / ${(P.total / 1e9).toFixed(1)} GB`
        : `${Number(P.done).toLocaleString()} / ${Number(P.total).toLocaleString()} ${esc(P.unit)}`);
      const rate = P.rate == null ? '' : (P.unit === 'bytes'
        ? `<span><b>${(P.rate / 1e6).toFixed(0)}</b> MB/s</span>`
        : `<span><b>${Number(P.rate).toFixed(0)}</b> seq/s</span>`);
      const batchFrac = (P.batch_total ? ((P.batch_done || 0) + (P.stage === 'scoring' && P.total ? P.done / P.total : 0)) / P.batch_total : 0);
      const batchPct = Math.max(0, Math.min(100, 100 * batchFrac));
      const bars = !P.total ? '' : `
        <div class="scorehead"><span>${esc(P.title || 'Scoring')}</span><b>${pct.toFixed(0)}%</b></div>
        <div class="bar score" title="${pct.toFixed(1)}%"><div style="width:${pct.toFixed(2)}%"></div></div>
        <div class="events"><span><b>${qty}</b></span>${rate}${P.eta_s != null ? `<span><b>${hms(P.eta_s)}</b> left</span>` : ''}</div>
        ${P.batch_total ? `<div class="scorehead"><span>Models in this batch</span><b>${(P.batch_done || 0)} / ${P.batch_total}</b></div>
        <div class="bar score" title="${batchPct.toFixed(1)}% of the batch"><div style="width:${batchPct.toFixed(2)}%"></div></div>` : ''}`;
      document.getElementById('now').innerHTML =
        `<div class="phase">${esc(CP0.phase || 'starting the merge search')}</div>` +
        bars +
        `<div class="sub">${esc(CP0.summary || 'Scores land on the four cards as each recipe finishes.')}</div>`;
    } else {
      document.getElementById('subline').textContent = `${D.run} · ${D.recipe}`;
      // now strip: phase, progress of the current step, next events
      const pct = D.steps ? Math.min(100, 100 * (R.progress_steps || 0) / D.steps) : 0;
      const ev = (R.next || []).map(e => `<span>${esc(e.what)} <b>${jst(e.t)}</b> <span style="color:var(--ink-3)">(in ${hms(e.t - Date.now()/1000)})</span></span>`).join('');
      document.getElementById('now').innerHTML =
        `<div class="phase">${esc(R.phase || '–')}</div>` +
        (R.detail ? `<div class="sub">${esc(R.detail)}</div>` : '') +
        `<div class="bar" title="${pct.toFixed(1)}% of ${D.steps} steps"><div style="width:${pct.toFixed(2)}%"></div></div>` +
        `<div class="events">${ev || '<span>no upcoming events</span>'}</div>` +
        (R.last_line ? `<div class="live">last log line ${jsts(R.last_line_t)} JST: ${esc(R.last_line)}</div>` : '');
    }
    const board = (D.campaign || {}).board || {};
    const hero = document.getElementById('hero');
    const SCR = D.screens || [], SCC = D.screen_comps || [], KN = D.king_names || {}, BAR = D.bars || {};
    const trainHero = SCC.length && D.status !== 'merging';
    const cards = [], grads = [];  // user (Oct 14): no merge board on the page
    const runCards = !trainHero ? '' : SCC.map(c => {
      const bar = BAR[c] ?? 0.002, rows = SCR.filter(r => r[c] && r[c].mu != null), ck = rows.filter(r => r.ckpt !== 'start');
      const st0 = rows.find(r => r.ckpt === 'start'), cur = ck.length ? ck[ck.length - 1] : null;
      const top = ck.length ? ck.reduce((a, b) => b[c].lcb > a[c].lcb ? b : a) : null;
      const e = cur ? cur[c] : (st0 ? st0[c] : null), won = top && top[c].lcb > bar;
      const who = cur ? `${esc(cur.ckpt)} · LCB ${sfmt(e.lcb, 5)} · bar ${bar}` + (st0 ? ` · ${sfmt(e.mu - st0[c].mu, 5)} vs start` : '')
        : (st0 ? `start merge (before training) · LCB ${sfmt(e.lcb, 5)} · bar ${bar}` : `no screen yet · bar ${bar}`);
      const how = won ? `✅ beats the king: ${esc(top.ckpt)} LCB ${sfmt(top[c].lcb, 5)} above the bar`
        : (top ? `best so far ${esc(top.ckpt)}: ${sfmt(top[c].mu, 5)} (LCB ${sfmt(top[c].lcb, 5)}); needs LCB above ${bar}`
          : `first checkpoint screen at step ${D.screen_every || D.save_every}, then every ${D.screen_every || D.save_every} steps`);
      return `<article class="compcard ${c.toLowerCase()}${won ? ' up' : ''}">
        <div class="k">${c} screen · vs ${esc(KN[c] || 'king')}</div>
        <div class="mu">${e ? sfmt(e.mu, 5) : '…'}</div>
        <div class="who">${who}</div><div class="how">${how}</div></article>`;
    }).join('') + (() => {
      if (!v0) return '';
      const prim = (D.kind || '').toLowerCase(), sp = k => (vl.val_split || {})[k], sp0 = k => (v0.val_split || {})[k];
      const one = (title, now, was, extra) => `<article class="compcard val">
        <div class="k">${title}</div><div class="mu">${vl.step > 0 ? sfmt(now - was, 5) : fmt(now, 5)}</div>
        <div class="who">${vl.step > 0 ? `${fmt(now, 5)} at step ${vl.step} · was ${fmt(was, 5)} at step 0 · lower is better` : 'step-0 reading (the start merge)'}</div>
        <div class="how">${extra}</div></article>`;
      const others = Object.keys(vl.val_split || {}).filter(k => k !== prim).map(k => `${k} ${vl.step > 0 ? sfmt(sp(k) - sp0(k), 5) : fmt(sp(k), 4)}`).join(' · ');
      const bestTxt = vbest && vbest.step > 0 ? `best ${fmt(vbest.val, 5)} at step ${vbest.step}` : `val every ${D.eval_every || '?'} steps`;
      return (sp(prim) != null && sp0(prim) != null ? one(`held-out CE · ${prim} split`, sp(prim), sp0(prim), 'EMA weights (what the checkpoints hold)') : '') +
        one('held-out CE · all sources', vl.val, v0.val, (others ? others + ' · ' : '') + bestTxt);
    })();
    const cardHtml = runCards + cards.map(c => {
      const up = c.mu != null && c.mu > 0;
      return `<article class="compcard ${esc(c.comp).toLowerCase()}${up ? ' up' : ''}">
        <div class="k">${esc(c.comp)} · vs ${esc(c.king || '?')}</div>
        <div class="mu">${c.mu == null ? '…' : sfmt(c.mu, 5)}</div>
        <div class="who">${c.mu == null ? 'waiting for the first screen' : (up ? 'better than the king' : 'still under the king')} · bar ${c.bar ?? '?'}${c.lcb != null ? ' · LCB ' + sfmt(c.lcb, 5) : ''}</div>
        <div class="how">${esc(c.method || '')}${c.desc ? ' — ' + esc(c.desc).slice(0, 140) : ''}</div>
      </article>`;
    }).join('');
    const ready = grads.filter(g => g.have).length;
    const gradHtml = !grads.length ? '' : `<div style="grid-column:1/-1"><div class="sub" style="margin:2px 0 6px">Previous-king gradients (the training step #n − #(n−1)). <b>${ready} on disk</b> · ${grads.length - ready} still downloading. Bright chips are #36–#40, the chain that cut loss before.</div><div class="gradrow">${grads.map(g => `<span class="gchip ${g.have ? (g.hot ? 'hot' : 'on') : ''}" title="${g.have ? 'both checkpoints are in, this step is in the search' : 'waiting for download'}">${esc(g.label)}</span>`).join('')}</div></div>`;
    hero.innerHTML = cardHtml + gradHtml;
    const tiles = [
      {k:'step', v: last ? `${last.step} / ${D.steps}` : `0 / ${D.steps}`, d: R.started ? `started ${jst(R.started)} JST · ${hms(D.rendered_epoch - R.started)} ago` : 'not started'},
      {k:'val CE now', v: vl ? fmt(vl.val) : '–', d: (vl && v0 && vl.step>0) ? `${sfmt(vl.val-v0.val)} vs step 0 (step ${vl.step})` + (vl.val_split && v0.val_split ? ` · ${Object.keys(vl.val_split).map(k => k + ' ' + sfmt(vl.val_split[k] - v0.val_split[k], 4)).join(' · ')}` : '') : 'step-0 reading only', cls: (vl && v0 && vl.step>0) ? (vl.val < v0.val ? 'good' : 'crit') : ''},
      {k:'best val', v: vbest ? fmt(vbest.val) : '–', d: vbest ? `step ${vbest.step} · ${sfmt(vbest.val - (v0?v0.val:vbest.val))}` : ''},
      {k:'train loss', v: last ? fmt(last.train) : '–', d: last ? `step ${last.step} · |g| ${fmt(last.gnorm,3)} · lr ${Number(last.lr).toExponential(1)}` : ''},
      {k:'throughput', v: R.tok_s ? `${(R.tok_s/1e3).toFixed(1)}k tok/s` : '–', d: R.step_s ? `${R.step_s.toFixed(0)} s/step (training only) · ${(D.tokens_per_step/1e6).toFixed(2)}M tok/step` : ''},
      {k:'finish (ETA)', v: R.finish ? jst(R.finish) : '–', d: R.finish ? `JST · in ${hms(R.finish - D.rendered_epoch)} incl. val/save/scoring` : ''},
      {k: D.primary ? `best ${D.primary} screen` : 'best screen', v: D.best_score ? sfmt(D.best_score.text_mu, 5) : '–', d: D.best_score ? `${D.best_score.ckpt} · ${D.primary ? 'bar' : 'target'} +${D.target}` + (D.best_score.lcb != null ? ` · LCB ${sfmt(D.best_score.lcb, 5)}` : '') : 'no checkpoint scored yet', cls: D.best_score ? ((D.best_score.lcb != null ? D.best_score.lcb > D.target : D.best_score.text_mu >= D.target) ? 'good' : '') : ''},
    ];
    document.getElementById('tiles').innerHTML = tiles.map(t => `<div class="tile"><div class="k">${t.k}</div><div class="v">${t.v}</div><div class="d ${t.cls||''}">${t.d||''}</div></div>`).join('');
    // charts: only redrawn when their data changed (keeps hover state between polls)
    const MK = (D.autopilot || {}).marks || [];  // autopilot LR-scale changes / rollbacks / anneals of this run
    const mkLegend = !MK.length ? '' : ' · autopilot: <span style="color:var(--good)">▲ raise</span> <span style="color:var(--warn)">▼ lower</span> ' +
      '<span style="color:var(--crit)">↶ rollback</span> <span style="color:#7a5cc2">◆ anneal · ↺ restore</span> (hover a marker for the reason)';
    const s = `${tr.length}|${vals.length}|${last?last.step:0}|${MK.length}`;
    if (s !== sig) {
      sig = s;
      document.getElementById('n-val').innerHTML = `mean token CE on ${esc(D.valset)}; dashed = step 0` + mkLegend;
      document.getElementById('n-train').textContent = `${D.train_every === 1 ? 'every optimizer step' : 'logged every ' + D.train_every + ' steps (mean over the window)'}; faint = raw, bold = 5-point mean`;
      document.getElementById('n-g').textContent = `global gradient norm before clipping (clip ${D.clip})`;
      document.getElementById('n-lr').innerHTML = esc(D.schedule) + (MK.length ? ' · logged lr = schedule × autopilot lr_scale' : '') + mkLegend;
      chart(document.getElementById('c-val'), [{name:'val CE', color:C.accent, pts: vals.map(v => ({x:v.step, y:v.val, extra: v0 ? `Δ vs step 0 ${sfmt(v.val - v0.val)}` + (v.t ? `<br>${jst(v.t)} JST` : '') : ''})), dots:true, area:true}], {label:'validation cross-entropy by step', xmax: D.steps, yref: v0 ? v0.val : null, dec: 4, marks: MK});
      const splitKeys = v0 && v0.val_split ? Object.keys(v0.val_split) : [];
      document.getElementById('w-split').style.display = splitKeys.length > 1 ? '' : 'none';
      if (splitKeys.length > 1) {
        const prim = (D.kind || '').toLowerCase(), pal = [C.accent, C.train, C.aux, '#7a5cc2'];
        const order = [...splitKeys.filter(k => k === prim), ...splitKeys.filter(k => k !== prim)];
        const dl = (v, k) => (v.val_split && v.val_split[k] != null) ? v.val_split[k] - v0.val_split[k] : null;
        document.getElementById('n-split').innerHTML = 'held-out CE of each source minus its step-0 value (below 0 = improved)' +
          order.map((k, i) => `<span class="lg" style="background:${pal[i % 4]}"></span>${esc(k)}`).join('') + '<span class="lg" style="background:var(--ink-3)"></span>all';
        const ser = order.map((k, i) => ({name: k, color: pal[i % 4], width: i ? 1.5 : 2.5, dots: !i, end: true,
          pts: vals.filter(v => dl(v, k) != null).map(v => ({x: v.step, y: dl(v, k), extra: i ? '' : order.slice(1).map(o => `${esc(o)}: ${sfmt(dl(v, o), 5)}`).join('<br>') + `<br>all: ${sfmt(v.val - v0.val, 5)}`}))}));
        ser.push({name: 'all', color: cs.getPropertyValue('--ink-3').trim() || '#8a94a3', width: 1.5, opacity: 0.8, pts: vals.map(v => ({x: v.step, y: v.val - v0.val}))});
        chart(document.getElementById('c-split'), ser, {label: 'validation cross-entropy change by source and step', xmax: D.steps, yref: 0, dec: 4, marks: MK});
      }
      const sm = tr.map((p,i) => { const w = tr.slice(Math.max(0,i-4), i+1); return {x:p.step, y: w.reduce((a,b)=>a+b.train,0)/w.length}; });
      chart(document.getElementById('c-train'), [{name:'train (5-pt mean)', color:C.train, pts: sm, end:true}, {name:'train', color:C.train, pts: tr.map(p=>({x:p.step,y:p.train})), width:1, opacity:0.35}], {label:'training loss by step', xmax: D.steps, dec: 4, marks: MK});
      chart(document.getElementById('c-tps'), [{name:'k tok/s', color:C.aux, pts: tr.filter(p => p.tps != null).map(p=>({x:p.step, y:p.tps/1e3, extra: p.t ? `${jst(p.t)} JST` : ''})), end:true}], {label:'throughput by step', xmax: D.steps, yzero:true, dec: 1});
      chart(document.getElementById('c-g'), [{name:'|g|', color:C.aux, pts: tr.map(p=>({x:p.step,y:p.gnorm})), end:true}], {label:'gradient norm by step', xmax: D.steps, yzero:true, dec: 3, marks: MK});
      chart(document.getElementById('c-lr'), [{name:'lr', color:C.aux, pts: tr.map(p=>({x:p.step, y:p.lr, extra: p.lr_scale != null ? `lr_scale ×${Number(p.lr_scale).toFixed(3)}` : ''})), end:true}], {label:'learning rate by step', xmax: D.steps, yzero:true, fmt: v => v.toExponential(1), marks: MK});
    }
    // this run's checkpoints
    const L = document.getElementById('ledger');
    if (SCC.length) {
      const P = D.primary;
      document.getElementById('ledger-title').textContent = `Checkpoints of this ${P} run vs the kings`;
      document.getElementById('ledger-note').textContent = `screens (first 4 800 sequences of sample A) of the checkpoint every ${D.screen_every || D.save_every} steps (saved every ${D.save_every}), paired vs the reigning king: ` +
        SCC.map(c => `${c} vs ${KN[c] || 'king'} (bar ${BAR[c]})`).join(', ') +
        `; start = the checkpoint this run started from (a merge: its campaign screen, 1 200 sequences); Δ ${P} = vs the start; ✅ = LCB above the bar (beats the king on the screen)`;
      const cell = (e, bar) => !e || e.mu == null ? '<td class="num">–</td>' :
        `<td class="num ${e.mu - bar > 0 ? 'pos' : (e.mu < 0 ? 'neg' : '')}" title="LCB ${sfmt(e.lcb)} · n ${e.n}">${sfmt(e.mu)} <small>(${sfmt(e.lcb)})</small>${e.lcb > bar ? ' ✅' : ''}</td>`;
      const vAt = r => { const n = r.ckpt === 'start' ? 0 : (r.ckpt === 'final' ? D.steps : parseInt(r.ckpt.replace(/\D/g, ''), 10)); return vals.find(v => v.step === n); };
      const vcell = r => { const v = vAt(r); return !v ? '<td class="num">–</td>' : `<td class="num ${v0 && v.step > 0 ? sgn(v0.val - v.val) : ''}">${fmt(v.val)}${v0 && v.step > 0 ? ` <small>(${sfmt(v.val - v0.val)})</small>` : ''}</td>`; };
      L.innerHTML = !SCR.length ? `<div class="empty">no checkpoint screened yet (every ${D.screen_every || D.save_every} steps)</div>` :
        '<table><thead><tr><th>checkpoint</th>' + SCC.map((c, i) => `<th>${c} vs ${esc(KN[c] || 'king')} (lcb)</th>` + (i ? '' : `<th>Δ ${c}</th>`)).join('') + '<th>val CE (Δ step 0)</th><th>scored JST</th></tr></thead><tbody>' +
        SCR.map(r => `<tr class="${r.ckpt === 'start' ? 'cur' : ''}"><td>${esc(r.ckpt)}</td>` + SCC.map((c, i) => cell(r[c], BAR[c]) + (i ? '' : `<td class="num ${sgn(r.d)}">${sfmt(r.d, 6)}</td>`)).join('') + `${vcell(r)}<td class="num">${jst(r.t)}</td></tr>`).join('') + '</tbody></table>';
    } else {
    const s0 = D.scores.find(x => x.ckpt === 'step-0' && x.text_mu != null);
    if (!D.scores.length) L.innerHTML = '<div class="empty">No checkpoint scored yet. Checkpoints are written every ' + D.save_every + ' steps and scored on the screen as they appear.</div>';
    else L.innerHTML = '<table><thead><tr><th>checkpoint</th><th>status</th><th>TEXT mu</th><th>Δ start</th><th>SE</th><th>LCB</th><th>text</th><th>math</th><th>code</th><th>n</th><th>note</th></tr></thead><tbody>' +
      D.scores.map(x => { const ok = x.text_mu != null && x.text_mu >= D.target; const cls = x.status==='done' ? (ok?'done':(x.text_mu!=null && x.text_mu < 0 ? 'below':'pending')) : x.status;
        const dd = (s0 && x.text_mu != null && x.ckpt !== 'step-0') ? x.text_mu - s0.text_mu : null;
        return `<tr><td>${x.ckpt}</td><td><span class="st ${cls}">${x.status==='done' ? (ok ? 'target met' : 'scored') : x.status}</span></td><td class="num">${sfmt(x.text_mu)}</td><td class="num ${sgn(dd)}">${sfmt(dd, 6)}</td><td class="num">${fmt(x.text_se)}</td><td class="num">${sfmt(x.lcb)}</td><td class="num">${sfmt(x.splits&&x.splits.text)}</td><td class="num">${sfmt(x.splits&&x.splits.math)}</td><td class="num">${sfmt(x.splits&&x.splits.code)}</td><td class="num">${x.n||'–'}</td><td class="l">${esc(x.note||'')}</td></tr>`; }).join('') + '</tbody></table>';
    }
    // evaluation: full duels per checkpoint
    const EV = document.getElementById('evals'), ER = D.evals || [];
    const tags = [...new Set(ER.flatMap(r => Object.keys(r.samples)))].sort();
    EV.innerHTML = !ER.length ? '<div class="empty">no duels scored yet</div>' : '<table><thead><tr><th>competition</th><th>vs king</th><th>checkpoint</th>' + tags.map(t => `<th>sample ${esc(t)}</th>`).join('') + '<th>mean (full)</th><th>passed</th><th>docs / windows (latest)</th><th>scored JST</th></tr></thead><tbody>' +
      ER.map(r => { const cells = tags.map(t => { const e = r.samples[t]; if (!e) return '<td class="num">–</td>';
          return `<td class="num ${e.full ? sgn(e.mu - (e.bar || 0.002)) : 'dim'}" title="LCB ${sfmt(e.lcb)} · n ${e.n}${e.full ? '' : ' (2400-seq screen)'}">${sfmt(e.mu)} ${e.full ? (e.accepted ? '✅' : '❌') : '<small>screen</small>'}</td>`; }).join('');
        const last = Object.values(r.samples).sort((a, b) => b.t - a.t)[0] || {};
        return `<tr><td>${esc(r.competition)}</td><td>${esc(r.king)}</td><td>${esc(r.run ? r.run + '/' : '')}<b>${esc(r.ckpt)}</b></td>${cells}<td class="num ${sgn(r.mean_full == null ? null : r.mean_full - (last.bar || 0.002))}">${sfmt(r.mean_full)}</td><td class="num">${r.n_full ? r.passed + ' / ' + r.n_full : '–'}</td><td class="num">${sfmt(last.docs)} / ${sfmt(last.windows)}</td><td class="num">${jst(r.t)}</td></tr>`; }).join('') + '</tbody></table>';
    // merge campaign: one row per recipe, a cell per competition vs its king
    const CP = D.campaign || {rows: [], kings: {}}, CR = CP.rows || [], comps = ['MAIN', 'MATH', 'CODE', 'TEXT'];
    const kb = c => (CP.kings || {})[c] || {};
    const best = {}; comps.forEach(c => { CR.forEach(r => { const e = r.cells[c]; if (e && e.mu != null && !(e.king && kb(c).king && e.king !== kb(c).king) && (best[c] == null || e.mu > best[c])) best[c] = e.mu; }); });
    document.getElementById('n-camp').innerHTML = '<b>green rows</b>: <span class="st done">submit X</span> = LCB above the bar against the reigning king of a competition we do not hold (★ = best LCB); <span class="st train">train base X</span> = no recipe passes there yet, best starting point to train for that king. ' +
      'screen = first 1 200 sequences of each competition\'s masked sample A for the public-model campaign (4 800 for training checkpoints), paired vs the reigning king; only verdicts against the reigning king are shown; ' +
      comps.map(c => `${c} vs <b>${esc(kb(c).king || '?')}</b> (bar ${kb(c).bar ?? '?'})`).join(' · ') +
      ' · green = mu above the bar, ✅ = LCB above the bar, <b>bold</b> = best in the column; hover a cell for LCB, splits, documents/windows' +
      (CP.phase ? `<br><span class="st busy">running</span> ${esc(CP.phase)} · ${CP.queued} recipe(s) not scored yet` : '') +
      (CP.summary ? `<br><b>${esc(CP.summary)}</b>` : '');
    const secCamp = document.getElementById('sec-camp'), secNow = document.getElementById('now');
    if (D.status === 'merging' && secNow) {
      if (hero.nextElementSibling !== secNow) hero.parentNode.insertBefore(secNow, hero.nextElementSibling);
      if (secCamp && secNow.nextElementSibling !== secCamp) secNow.parentNode.insertBefore(secCamp, secNow.nextElementSibling);
    } else if (secCamp) {  // a training run keeps its own result on top: the merge table goes back under the duel panel
      const anchor = trainHero ? document.getElementById('evals').parentNode : hero;
      if (anchor.nextElementSibling !== secCamp) anchor.parentNode.insertBefore(secCamp, anchor.nextElementSibling);
    }
    // Hide charts only when there is nothing to plot. Keep the phase strip + tiles
    // visible during FSDP init / step-0 val (status busy|running) so a live trainer
    // does not look "missing" just because train_log.jsonl is not written yet.
    const quietTrain = !(D.train || []).length && !(D.val || []).length;
    const liveBoot = quietTrain && (D.status === 'busy' || D.status === 'running' || D.status === 'merging');
    const charts = document.querySelector('.charts');
    if (charts) charts.style.display = quietTrain ? 'none' : '';
    if (secNow) secNow.style.display = (quietTrain && !liveBoot) ? 'none' : '';
    const tilesEl = document.getElementById('tiles');
    if (tilesEl) tilesEl.style.display = (quietTrain && !liveBoot) ? 'none' : '';
    const hiddenNote = CP.hidden ? `<div class="sub" style="margin:4px 0 8px">${CP.hidden} recipe${CP.hidden === 1 ? '' : 's'} below every king and outside the top 8 of each competition are hidden (results stay in eval_cache/results).</div>` : '';
    document.getElementById('camp').innerHTML = !CR.length ? '<div class="empty">no merge campaign yet</div>' : hiddenNote +
      '<table><thead><tr><th>recipe</th><th>verdict</th><th>method</th><th>what it is</th>' + comps.map(c => `<th>${c} vs ${esc(kb(c).king || '?')}</th>`).join('') + '<th>status</th></tr></thead><tbody>' +
      CR.map(r => { const cells = comps.map(c => { const e = r.cells[c]; if (!e || e.mu == null) return '<td class="num">–</td>';
          const stale = e.king && kb(c).king && e.king !== kb(c).king;  // scored against an earlier king
          const over = e.mu - (e.bar || 0), cls = stale ? 'dim' : (over > 0 ? 'pos' : (e.mu < 0 ? 'neg' : ''));
          const tt = `vs ${e.king || '?'} · LCB ${sfmt(e.lcb)} · bar ${e.bar} · n ${e.n} · ` + Object.entries(e.src || {}).map(([k, x]) => `${k} ${sfmt(x)}`).join(' · ') + (e.docs != null ? ` · docs ${sfmt(e.docs)} · windows ${sfmt(e.windows)}` : '');
          const txt = sfmt(e.mu) + (e.pass ? ' ✅' : '');
          const mag = Math.min(0.55, Math.abs(e.mu) / 0.002);
          const heat = stale || !mag ? '' : ` style="background:${e.mu >= 0 ? 'rgba(26,127,55,' + mag.toFixed(2) + ')' : 'rgba(192,57,43,' + mag.toFixed(2) + ')'}"`;
          return `<td class="num ${cls}"${heat} title="${esc(tt)}">${e.mu === best[c] && !stale ? '<b>' + txt + '</b>' : txt}${stale ? ' <small>vs ' + esc(e.king) + '</small>' : ''}</td>`; }).join('');
        const st = r.status === 'scored' ? `<span class="st done">scored ${jst(r.t)}</span>` : `<span class="st ${r.status === 'scoring' ? 'busy' : 'pending'}">${esc(r.status)}</span>`;
        const marks = (r.marks || []).map(m => `<span class="st ${m.startsWith('submit') ? 'done' : 'train'}">${esc(m)}</span>`).join(' ');
        return `<tr class="${(r.marks || []).length ? 'cur' : ''}"><td><b>${esc(r.label)}</b></td><td>${marks || '<span style="color:var(--ink-3)">–</span>'}</td><td>${esc(r.mode || 'linear')}</td><td class="l">${esc(r.desc || '')}</td>${cells}<td>${st}</td></tr>`; }).join('') + '</tbody></table>';
    // eval jobs
    const EJ = document.getElementById('evaljobs'), EJS = D.eval_jobs || [];
    EJ.innerHTML = !EJS.length ? '<div class="empty">no eval jobs in the last 6 h</div>' : '<table><thead><tr><th>job</th><th>progress</th><th>detail</th><th>status</th></tr></thead><tbody>' +
      EJS.map(j => `<tr><td>${esc(j.name)}</td><td class="num">${j.frac != null ? `<span class="meter"><i style="width:${(100*j.frac).toFixed(1)}%"></i></span>${(100*j.frac).toFixed(0)}%` : '–'}</td><td class="l">${esc(j.detail || '')}</td><td><span class="st ${j.status === 'running' ? 'busy' : 'done'}">${esc(j.status)}</span></td></tr>`).join('') + '</tbody></table>';
    // validator duel of our submission
    const VE = document.getElementById('vduel'), VD = D.validator || {};
    let vh = `<div class="vline">TEXT king: <b>${esc(VD.king || '?')}</b> · our uid ${esc(VD.uid)}</div>`;
    if (VD.running) { const r = VD.running;
      vh += `<div class="vline"><span class="st busy">running</span> challenge ${esc(r.challenge_id || '')} · stage <b>${esc(r.stage || '')}</b> · ${r.percent != null ? Number(r.percent).toFixed(0) : 0}% (${r.progress || 0} / ${r.total || '?'}) · provisional mu ${sfmt(r.provisional_mu_hat)} lcb ${sfmt(r.provisional_lcb)} · ${Math.round((r.elapsed_seconds || 0) / 60)} min</div>`;
    } else if ((VD.queue || []).length) vh += `<div class="vline"><span class="st pending">${esc(VD.queue.join(' · '))}</span> in the TEXT queue</div>`;
    else if (!(VD.history || []).length) vh += '<div class="empty">no submission seen</div>';
    vh += (VD.history || []).map(h => `<div class="vline">${jst(Date.parse(h.timestamp) / 1000)} uid ${esc(h.uid)} · challenge ${esc(h.challenge_id || '')}: <span class="st ${h.accepted ? 'done' : (h.verdict === 'error' ? 'crit' : 'below')}">${esc(h.verdict || '')}</span> mu ${sfmt(h.mu_hat)} lcb ${sfmt(h.lcb)} ${esc(h.error_code || '')}</div>`).join('');
    VE.innerHTML = vh;
    // all runs
    const RU = document.getElementById('runs');
    RU.innerHTML = !D.runs.length ? '<div class="empty">no runs</div>' : '<table><thead><tr><th>run</th><th>started JST</th><th>recipe</th><th>steps</th><th>start vs T1</th><th>val Δ</th><th>scored</th><th>best Δ start</th><th>status</th></tr></thead><tbody>' +
      D.runs.map(r => `<tr class="${r.name === D.run ? 'cur' : ''}"><td>${esc(r.name)}</td><td class="num">${jst(r.started)}</td><td class="l">${esc(r.recipe)}</td><td class="num">${r.last_step ?? '–'}${r.steps ? ' / ' + r.steps : ''}</td><td class="num">${sfmt(r.start_mu)}</td><td class="num ${sgn(r.val_delta == null ? null : -r.val_delta)}">${sfmt(r.val_delta)}${r.val_step ? ' @' + r.val_step : ''}</td><td class="num">${esc(r.scored || '–')}</td><td class="num ${sgn(r.best_delta)}">${sfmt(r.best_delta, 6)}${r.best_ckpt ? ' (' + r.best_ckpt + ')' : ''}</td><td><span class="st ${r.status_cls}">${esc(r.status)}</span></td></tr>`).join('') + '</tbody></table>';
    // GPUs
    const G = document.getElementById('gpus');
    G.innerHTML = !D.gpus.length ? '<div class="empty">nvidia-smi unavailable</div>' : '<table><thead><tr><th>GPU</th><th>util</th><th>memory</th><th>power</th><th>temp</th></tr></thead><tbody>' +
      D.gpus.map(g => `<tr><td>${g.index} · ${esc(g.name)}</td><td class="num"><span class="meter"><i style="width:${g.util}%"></i></span>${g.util}%</td><td class="num">${(g.mem/1024).toFixed(1)} / ${(g.mem_total/1024).toFixed(0)} GiB</td><td class="num">${g.power != null ? g.power.toFixed(0) + ' W' : '–'}</td><td class="num">${g.temp != null ? g.temp + '°C' : '–'}</td></tr>`).join('') + '</tbody></table>';
    // background jobs
    const J = document.getElementById('jobs');
    J.innerHTML = !D.jobs.length ? '<div class="empty">none in the last 12 h</div>' : '<table><thead><tr><th>job</th><th>progress</th><th>detail</th><th>status</th></tr></thead><tbody>' +
      D.jobs.map(j => `<tr><td>${esc(j.name)}</td><td class="num">${j.frac != null ? `<span class="meter"><i style="width:${(100*j.frac).toFixed(1)}%"></i></span>${(100*j.frac).toFixed(0)}%` : '–'}</td><td class="l">${esc(j.detail || '')}</td><td><span class="st ${j.status_cls}">${esc(j.status)}</span></td></tr>`).join('') + '</tbody></table>';
    renderAutopilot();
    document.getElementById('foot').innerHTML = `<span>${esc(D.hardware)}</span><span>pool ${esc(D.pool)}</span><span>val ${esc(D.valset)}</span><span>${esc(D.log_path)}</span><span>rendered ${jsts(D.rendered_epoch)} JST</span>`;
    showAge();
  }
  // autopilot (scripts/train/autopilot.py): staged candidates + one card per competition; on top while a status is < 1 h old
  function renderAutopilot() {
    const AP = D.autopilot || {}, cards = AP.cards || [], ready = AP.ready || [], now = D.rendered_epoch;
    const sec = document.getElementById('sec-ap');
    if (!cards.length && !ready.length) { sec.style.display = 'none'; return; }
    sec.style.display = '';
    const top = AP.fresh || ready.some(r => r.t && now - r.t < 86400 && !r.former);
    const anchor = top ? document.getElementById('subline') : document.getElementById('ledger').parentNode;
    if (anchor.nextElementSibling !== sec) anchor.parentNode.insertBefore(sec, anchor.nextElementSibling);
    const newest = cards.filter(c => c.t && c.kings).sort((a, b) => b.t - a.t)[0], KG = (newest || {}).kings || {};
    const kl = ['MAIN', 'MATH', 'CODE', 'TEXT'].filter(c => KG[c]).map(c => `${c} #${esc(KG[c].reign ?? '?')} ${esc(KG[c].repo || KG[c].digest)}`).join(' · ');
    document.getElementById('n-ap').innerHTML = 'one controller per competition: trains, steers the LR live from the paired EMA-val trend, screens every checkpoint on all four competitions vs the reigning kings, confirms on samples B/C and stages winners; submitting stays manual' +
      (kl ? ` · reigning kings: ${kl}` : '') + (top ? '' : ' · <b>no controller wrote a status in the last hour</b> (shown below the training sections)') + ` · ${esc(AP.dir)}`;
    const full = f => Object.entries(f || {}).map(([k, v]) => `${esc(k.replace('tag:', ''))} ${sfmt(v.mu)} (LCB ${sfmt(v.lcb)})`).join(' · ');
    document.getElementById('ap-ready').innerHTML = ready.map(r => `<div class="ready${r.former ? ' former' : ''}">
      <div class="big">${r.former ? 'staged earlier (the king has changed since)' : 'SUBMIT-READY'} · ${esc(r.comp)} vs ${esc(r.king || '?')} · bar ${esc(r.bar ?? '?')}</div>
      <div>full samples: <b>${full(r.full) || '–'}</b> · <span class="st ${esc(r.stage.cls)}">${esc(r.stage.state)}</span> · ${when(r.t)}</div>
      <div>staged dir <span class="mono">${esc(r.staged_dir || '–')}</span>${r.staged_dir && !r.staged_exists ? ' <span class="st pending">not on disk yet</span>' : ''}</div>
      <div>stage log <span class="mono">${esc(r.stage_log || '–')}</span>${r.stage.line ? ` · last line: <span class="mono">${esc(r.stage.line)}</span>` : ''}</div>
      <div>checkpoint <span class="mono">${esc(r.path || '–')}</span> · check the stage log, then submit by hand (one submission per hotkey)</div></div>`).join('');
    const PH = {train: 'training', screen: 'screening', merge: 'merge round', needs_attention: 'needs attention', starting: 'starting'};
    const ACT = {raise: 'done', keep: 'idle', lower: 'busy', rollback: 'crit', plateau: 'busy'};
    const EVK = {SUBMIT_READY: 'hot', confirmed: 'hot', give_up: 'bad', screen_failed: 'bad', confirm_failed: 'bad', merge_failed: 'bad', recover: 'bad', disk_low: 'bad'};
    const lrf = x => x == null || isNaN(x) ? '–' : Number(x).toExponential(2), scf = x => x == null || isNaN(x) ? '?' : Number(x).toFixed(3);
    document.getElementById('ap').innerHTML = cards.map(c => {
      if (c.error) return `<article class="apcard"><div class="hd"><span class="c">${esc(c.comp)}</span><span class="st crit">status not readable</span></div><div class="kv">${esc(c.error)}</div></article>`;
      const age = now - c.t, L = c.lr || {}, S = c.screen, ph = PH[c.phase] ? c.phase : 'starting';
      const hd = `<div class="hd"><span class="c">${esc(c.comp)}</span><span class="pill ap-${ph}">${esc(PH[c.phase] || c.phase)}</span>` +
        `<span class="live${age > 600 ? ' stale' : ''}">status ${hms(age)} old${age > 600 ? ' · controller silent?' : ''}</span></div>`;
      const runl = `<div class="kv"><div>run <b>${esc(c.run || '–')}</b>${c.run && c.run === D.run ? ' <span class="st done">charts below</span>' : ''}</div><div>start <span class="mono">${esc(c.start || '–')}</span></div></div>`;
      const lrl = [`lr <b>${L.base != null ? lrf(L.base) : '?'}</b> × scale <b>${scf(L.scale)}</b> = <b>${lrf(L.eff)}</b>${L.dense && L.dense !== 1 ? ` (dense ×${esc(L.dense)})` : ''} · ceiling <b>${L.ceiling != null ? scf(L.ceiling) : 'none'}</b>` +
          (L.anneal ? ` · <span class="st busy">annealing ${esc(L.anneal)} steps from step ${esc(L.anneal_from ?? '?')}</span>` : ''),
        (L.now != null ? `trainer at step <b>${esc(L.now_step)}</b>: lr <b>${lrf(L.now)}</b> (schedule × ${scf(L.now_scale)})` + (L.val != null ? ` · EMA val <b>${fmt(L.val)}</b> @${esc(L.val_step)}` : '') + ' · ' : '') +
          (L.best_val != null ? `best EMA val <b>${fmt(L.best_val)}</b> at step ${esc(L.best_step)}` : 'no best val yet'),
        L.last ? `last decision, step <b>${esc(L.last.step)}</b>: <span class="st ${ACT[L.last.act] || 'idle'}">${esc(L.last.act)}</span> ${esc(L.last.why || '')}` : 'no LR decision yet',
        ((L.history || []).length ? `changes (${esc(L.n_changes)}): ` + L.history.map(h => `step ${esc(h.step)} ${esc(h.act)} ×${scf(h.from)}→×${scf(h.to)}`).join(' · ') : 'no LR change this segment') + ` · trend judged since step ${esc(L.since ?? 0)}`];
      const CT = L.control, AK = L.ack;
      if (CT && AK && CT.id !== AK.id) lrl.push(`<span class="st busy">CONTROL id ${esc(CT.id)} not applied yet</span> (trainer acknowledged id ${esc(AK.id)} at step ${esc(AK.step)})`);
      else if (AK) lrl.push(`trainer applied CONTROL id ${esc(AK.id)} at step ${esc(AK.step)}: lr ${lrf(AK.lr)}, ×${scf(AK.lr_scale)}${AK.anneal ? ', anneal ' + esc(AK.anneal) : ''}${AK.note ? ' (' + esc(AK.note) + ')' : ''}`);
      let scr = '<h3>latest screen</h3>';
      if (!S) scr += '<div class="empty">no screen yet</div>';
      else if (S.skipped || S.failed) scr += `<div class="kv"><div>${esc(S.run)}/${esc(S.ckpt)}: <span class="st crit">${S.skipped ? 'skipped' : 'failed'}</span> ${esc(S.skipped || S.failed)}</div></div>`;
      else scr += `<div class="kv"><div><b>${esc(S.run)}/${esc(S.ckpt)}</b> (step ${esc(S.step)}${S.branch ? ' · branch ' + esc(S.branch) : ''}${S.annealed ? ' · annealed weights' : ''}) · ${when(S.t)}</div></div>` +
        '<table><thead><tr><th>comp</th><th>mu</th><th>lcb</th><th>vs king</th><th>bar</th><th>gap to bar</th></tr></thead><tbody>' +
        S.rows.map(r => `<tr class="${r.mu != null && r.mu >= r.bar ? 'cur' : ''}${r.comp === c.comp ? ' own' : ''}"><td>${esc(r.comp)}</td><td class="num ${sgn(r.mu)}">${sfmt(r.mu)}</td>` +
          `<td class="num" title="n ${esc(r.n)}">${sfmt(r.lcb)}${r.lcb != null && r.lcb > r.bar ? ' ✅' : ''}</td><td>${esc(r.king || '–')}</td><td class="num">${esc(r.bar)}</td><td class="num ${sgn(r.gap)}">${sfmt(r.gap)}</td></tr>`).join('') + '</tbody></table>';
      const B = c.best || [];
      const best = '<h3>best ever per competition</h3>' + (!B.length ? '<div class="empty">nothing screened yet</div>' :
        '<table><thead><tr><th>comp</th><th>mu</th><th>lcb</th><th>checkpoint</th><th>vs king</th><th>screened</th></tr></thead><tbody>' +
        B.map(b => `<tr class="${b.mu != null && b.mu >= b.bar && !b.former ? 'cur' : ''}${b.comp === c.comp ? ' own' : ''}"><td>${esc(b.comp)}</td><td class="num ${b.former ? 'dim' : sgn(b.mu)}">${sfmt(b.mu)}</td>` +
          `<td class="num${b.former ? ' dim' : ''}">${sfmt(b.lcb)}</td><td title="${esc(b.path)}">${esc(b.run)}/${esc(b.ckpt)}</td><td>${esc(b.king || '?')}${b.former ? ' <small>(former king)</small>' : ''}</td><td class="num">${when(b.t)}</td></tr>`).join('') + '</tbody></table>');
      const EV = c.events || [];
      const evs = '<h3>last events (newest first)</h3>' + (!EV.length ? '<div class="empty">none</div>' : '<div class="evlist">' + EV.map(e => {
        const txt = Object.entries(e).filter(([k]) => k !== 't' && k !== 'kind').map(([k, v]) => `${k}=${v !== null && typeof v === 'object' ? JSON.stringify(v) : v}`).join(' ');
        return `<div class="${EVK[e.kind] || ''}" title="${esc(txt)}"><span class="t">${when(e.t)}</span><b>${esc(e.kind)}</b> ${esc(txt.length > 220 ? txt.slice(0, 220) + '…' : txt)}</div>`; }).join('') + '</div>');
      return `<article class="apcard${age > 3600 ? ' old' : ''}">${hd}${runl}<h3>learning rate</h3><div class="kv">${lrl.map(x => `<div>${x}</div>`).join('')}</div>${scr}${best}${evs}</article>`;
    }).join('');
  }
  // freshness: the data is re-rendered on the box every few seconds and polled from data.json every 10 s
  function showAge() {
    const age = Date.now() / 1000 - D.rendered_epoch, el = document.getElementById('live');
    el.textContent = `data ${hms(age)} old · updated ${jsts(D.rendered_epoch)} JST · polling every 10 s`;
    el.className = 'live' + (age > 90 ? ' stale' : '');
  }
  let fails = 0;
  async function poll() {
    if (document.hidden) return;
    try {
      const r = await fetch('data.json?' + Date.now(), {cache: 'no-store'});
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const nd = await r.json(); fails = 0;
      if (nd.rendered_epoch > D.rendered_epoch) { D = nd; render(); }
    } catch (e) {  // a dropped request (noisy tunnel): retry in 2 s, and reload the whole page after 3 failures in a row
      if (++fails >= 3) { fails = 0; location.reload(); } else setTimeout(poll, 2000);
    }
  }
  setInterval(poll, 10000); setInterval(showAge, 1000);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) poll(); });

  // --- charts (inline SVG, one scale per axis)
  function chart(el, series, opts) {
    const W = Math.max(280, Math.round(el.clientWidth || 900)), H = Math.round(Math.max(260, Math.min(470, W * 0.46)));
    const m = {l: 78, r: 20, t: 18, b: 38};
    const pts = series.flatMap(s => s.pts);
    if (!pts.length) { el.innerHTML = '<div class="empty">no data yet</div>'; return; }
    // x spans the data (+12 % headroom, rounded), not the whole planned run: 60 steps of an 800-step plan used to fill 8 % of the width
    const xdata = Math.max(...pts.map(p => p.x)), xspan = xdata * 1.12 + 4, xq = Math.pow(10, Math.floor(Math.log10(xspan))) / 2;
    const xmax = Math.min(Math.max(opts.xmax || 0, xdata), Math.ceil(xspan / xq) * xq);
    let ymin = Math.min(...pts.map(p => p.y), ...(opts.yref!=null?[opts.yref]:[])), ymax = Math.max(...pts.map(p => p.y), ...(opts.yref!=null?[opts.yref]:[]));
    if (opts.yzero) ymin = Math.min(0, ymin);
    if (ymax - ymin < 1e-9) { ymax += 1e-3; ymin -= 1e-3; }
    const pad = (ymax - ymin) * 0.08; ymin -= pad; ymax += pad;
    const sx = x => m.l + (x / (xmax || 1)) * (W - m.l - m.r);
    const sy = y => m.t + (1 - (y - ymin) / (ymax - ymin)) * (H - m.t - m.b);
    const yt = ticks(ymin, ymax, Math.max(5, Math.round(H / 60))), xt = ticks(0, xmax, Math.max(6, Math.round(W / 110)));
    const dec = Math.max(opts.dec != null ? opts.dec : 0, tickDec(yt));  // enough decimals that no two tick labels read the same
    let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${opts.label}">`;
    g += '<g class="grid">' + yt.map(t => `<line x1="${m.l}" x2="${W-m.r}" y1="${sy(t)}" y2="${sy(t)}"/>`).join('') + '</g>';
    g += '<g class="axis">' + `<line x1="${m.l}" x2="${m.l}" y1="${m.t}" y2="${H-m.b}"/><line x1="${m.l}" x2="${W-m.r}" y1="${H-m.b}" y2="${H-m.b}"/>` + '</g>';
    g += yt.map(t => `<text x="${m.l-8}" y="${sy(t)+4.5}" text-anchor="end">${opts.fmt ? opts.fmt(t) : t.toFixed(dec)}</text>`).join('');
    g += xt.map(t => `<text x="${sx(t)}" y="${H-m.b+19}" text-anchor="middle">${Math.round(t)}</text>`).join('');
    g += `<text class="lbl" x="${W-m.r}" y="${H-2}" text-anchor="end">step</text>`;
    if (opts.yref != null) g += `<line x1="${m.l}" x2="${W-m.r}" y1="${sy(opts.yref)}" y2="${sy(opts.yref)}" stroke="var(--ink-3)" stroke-dasharray="4 4" stroke-width="1"/>`;
    const MKC = {raise: 'var(--good)', lower: 'var(--warn)', rollback: 'var(--crit)', anneal: '#7a5cc2', restore: '#7a5cc2', scale: 'var(--ink-3)'};
    const MKG = {raise: '▲', lower: '▼', rollback: '↶', anneal: '◆', restore: '↺', scale: '•'};
    const marks = (opts.marks || []).filter(k => k.step >= 0 && k.step <= xmax);
    g += marks.map(k => { const x = sx(k.step).toFixed(1), col = MKC[k.kind] || MKC.scale;
      return `<line x1="${x}" x2="${x}" y1="${m.t}" y2="${H-m.b}" stroke="${col}" stroke-width="1.5" stroke-dasharray="4 4" opacity="0.85"/>` +
        `<text x="${x}" y="${m.t + 2}" text-anchor="middle" style="fill:${col};font-size:15px">${MKG[k.kind] || '•'}</text>`; }).join('');
    for (const s of series) {
      if (!s.pts.length) continue;
      const d = s.pts.map((p,i) => (i?'L':'M') + sx(p.x).toFixed(1) + ' ' + sy(p.y).toFixed(1)).join(' ');
      if (s.area) g += `<path d="${d} L ${sx(s.pts[s.pts.length-1].x).toFixed(1)} ${H-m.b} L ${sx(s.pts[0].x).toFixed(1)} ${H-m.b} Z" fill="${s.color}" opacity="0.08"/>`;
      g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${(s.width||2) * 1.35}" opacity="${s.opacity||1}" stroke-linejoin="round" stroke-linecap="round"/>`;
      if (s.dots) g += s.pts.map(p => `<circle cx="${sx(p.x)}" cy="${sy(p.y)}" r="4.6" fill="${s.color}" stroke="var(--surface)" stroke-width="1.8"/>`).join('');
      const e = s.pts[s.pts.length-1];
      if (s.end) g += `<circle cx="${sx(e.x)}" cy="${sy(e.y)}" r="5.5" fill="${s.color}" stroke="var(--surface)" stroke-width="2"/>`;
    }
    g += `<rect class="hit" x="${m.l}" y="${m.t}" width="${W-m.l-m.r}" height="${H-m.t-m.b}" fill="transparent"/><line class="xh" x1="0" x2="0" y1="${m.t}" y2="${H-m.b}" stroke="var(--ink-3)" stroke-width="1" style="display:none"/></svg>`;
    el.innerHTML = g;
    const svg = el.querySelector('svg'), hit = svg.querySelector('.hit'), xh = svg.querySelector('.xh');
    const main = series[0];
    if (!main.pts.length) return;
    hit.addEventListener('mousemove', ev => {
      const r = svg.getBoundingClientRect(); const x = (ev.clientX - r.left) * W / r.width;
      const xv = (x - m.l) / (W - m.l - m.r) * xmax;
      let best = main.pts[0]; for (const p of main.pts) if (Math.abs(p.x - xv) < Math.abs(best.x - xv)) best = p;
      xh.setAttribute('x1', sx(best.x)); xh.setAttribute('x2', sx(best.x)); xh.style.display = '';
      tip.style.display = 'block'; tip.style.left = (ev.clientX + 14) + 'px'; tip.style.top = (ev.clientY + 14) + 'px';
      const near = marks.filter(k => Math.abs(sx(k.step) - x) < 10);
      tip.innerHTML = `step ${best.x}<br>${main.name}: ${opts.fmt ? opts.fmt(best.y) : best.y.toFixed(dec)}` + (best.extra ? `<br>${best.extra}` : '') +
        near.map(k => { const w = String(k.why || ''); return `<br><b>step ${k.step}: ${esc(k.label)}</b>` + (w ? `<br>${esc(w.length > 110 ? w.slice(0, 110) + '…' : w)}` : ''); }).join('');
    });
    hit.addEventListener('mouseleave', () => { tip.style.display = 'none'; xh.style.display = 'none'; });
  }
  function ticks(a, b, n) { const span = b - a || 1; const raw = span / n; const p = Math.pow(10, Math.floor(Math.log10(raw))); const f = raw / p; const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p; const out = []; for (let t = Math.ceil(a / step) * step; t <= b + 1e-12; t += step) out.push(+t.toFixed(10)); return out; }
  function tickDec(t) { if (t.length < 2) return 3; const s = t[1] - t[0]; return Math.max(0, Math.min(6, -Math.floor(Math.log10(s)))); }
  // redraw when a chart's box changes width (window resize, or the layout settling after other panels fill in): the SVG
  // is drawn at its on-screen width, so a stale one would be scaled and its text shrunk
  let resized;
  const redraw = () => { clearTimeout(resized); resized = setTimeout(() => {
    const stale = [...document.querySelectorAll('.chart svg')].some(svg => {
      const w = svg.parentNode.clientWidth, vb = +(svg.getAttribute('viewBox') || '').split(' ')[2];
      return w > 0 && Math.abs(w - vb) > 2; });
    if (stale) { sig = ''; render(); } }, 150); };
  if (window.ResizeObserver) { const ro = new ResizeObserver(redraw); document.querySelectorAll('.chart').forEach(c => ro.observe(c)); }
  window.addEventListener('resize', redraw);
  render();
})();
</script>
"""


# ----------------------------------------------------------------------------- sources on the box

def _seq_len(a: dict) -> int:
    """Tokens per training row: the pool's seq_len (8 192 for the long-context pools), else 2 048."""
    try:
        first = str(a.get("pool", "")).split(",")[0]
        meta = json.loads((Path(first) / "meta.json").read_text())
        return int(meta.get("seq_len", 2048))
    except Exception:
        return 2048

def trainer_args(name: str) -> dict[str, str]:
    """cpt_fsdp.py flags from the pm2 launcher script (later flags win, as in argparse); {} if there is none."""
    p = LOGS / f"run_{name}.sh"
    if not p.exists():
        return {}
    text = p.read_text().replace("\\\n", " ")
    m = re.search(r"--nproc_per_node\s+(\d+)", text)
    out = {"_world": m.group(1) if m else "8"}
    m = re.search(r"cpt_fsdp\.py(.*)$", text, re.S)
    toks = shlex.split(m.group(1)) if m else []
    i = 0
    while i < len(toks):
        if toks[i].startswith("--"):
            if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                out[toks[i][2:]] = toks[i + 1]
                i += 2
                continue
            out[toks[i][2:]] = "1"
        i += 1
    return out


def trained_part(a: dict, n_layers: int = 45) -> str:
    """'full-parameter' or which decoder layers --freeze leaves trainable (routers are always frozen)."""
    try:
        freeze = re.compile(a.get("freeze") or r"\.mlp\.gate\.")
    except re.error:
        return "partial (custom --freeze)"
    trained = [i for i in range(n_layers) if not freeze.search(f"model.layers.{i}.self_attn.o_proj.weight")]
    if len(trained) == n_layers and not freeze.search("model.embed_tokens.weight"):
        return "full-parameter"
    span = f"layers {trained[0]}–{trained[-1]}" if trained and trained == list(range(trained[0], trained[-1] + 1)) else f"{len(trained)} layers"
    return f"{span} + lm_head trainable" if not freeze.search("lm_head.weight") else span + " trainable"


def describe(a: dict) -> tuple[str, str]:
    """(one-line recipe, lr schedule) from trainer flags."""
    if not a:
        return "recipe unknown (no launcher script)", "learning rate"
    frac, warm, lr = float(a.get("min-lr-frac", 0.1)), a.get("warmup", "50"), a.get("lr", "1e-5")
    dense = float(a.get("lr-dense-frac", 1.0))
    if dense != 1.0:  # experts at --lr, attention/shared/dense/head at lr × --lr-dense-frac
        lr = f"experts {lr} / dense {float(lr) * dense:.1e}".replace("e-0", "e-")
    sched = f"lr {lr} constant after {warm} warmup steps" if frac >= 1 else f"lr {lr}, warmup {warm}, cosine to {frac:g}× peak"
    if float(a.get("ema", 0)) > 0:
        sched += f" · EMA {a['ema']}/step (val + checkpoints = EMA weights)"
    if int(a.get("lora-r", 0)) > 0:
        mode = f"LoRA r{a['lora-r']} α{a.get('lora-alpha', '32')} on {a.get('lora-targets', 'attn,shared,dense,experts')}"
    else:
        mode = f"{trained_part(a)} ({a.get('dtype', 'fp32')} masters, {a.get('optim', 'adamw8bit')}" + (", optimizer offloaded" if a.get("offload-opt") == "1" else "") + ")"
    b, k, w = int(a.get("batch", 2)), int(a.get("accum", 16)), int(a.get("_world", 8))
    return f"{mode} · {sched} · {b * k * w} seq/step ({b}×{k}×{w}) · {a.get('steps', '800')} steps", sched


def cmd(args: list[str], timeout: float = 15.0) -> str:
    """stdout of a command, or '' after `timeout`. A child stuck in uninterruptible I/O (D state, e.g. on the container's
    FUSE /proc) cannot be killed or reaped, so it is abandoned instead of waited for: the render never blocks."""
    try:
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    except OSError:
        return ""
    try:
        return p.communicate(timeout=timeout)[0]
    except subprocess.TimeoutExpired:
        p.kill()
        return ""


def pm2_status() -> dict[str, str]:
    """Run name -> 'online' for every live cpt_fsdp.py (--out <ckpt dir>). From pgrep, not `pm2 jlist`: pm2 reads
    per-process stats through the container's FUSE /proc and hung there in D state on this box (2026-10-09)."""
    alive = set(re.findall(r"--out /data/teutonic/ckpt/([^\s/]+)", cmd(["pgrep", "-af", "cpt_fsdp.py"])))
    return {name: "online" for name in alive}


def log_lines(name: str) -> list[tuple[float, str]]:
    p = LOGS / f"pm2_{name}.out.log"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(errors="replace").splitlines():
        m = PM2_LINE.match(line)
        if m:  # pm2 --time stamps are the box's local time (UTC here); timestamp() treats naive as local
            out.append((dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S").timestamp(), m.group(2)))
    return out


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass  # a line being written right now
    return rows


def gpu_stats() -> list[dict]:
    out = cmd(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
               "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        num = lambda s: float(s) if re.match(r"^[\d.]+$", s) else None  # noqa: E731
        gpus.append({"index": int(f[0]), "name": f[1].replace("NVIDIA ", ""), "util": int(num(f[2]) or 0), "mem": num(f[3]) or 0,
                     "mem_total": num(f[4]) or 0, "power": num(f[5]), "temp": int(num(f[6])) if num(f[6]) is not None else None})
    return gpus


def background_jobs() -> list[dict]:
    """King downloads (download_<digest8>.log) and merge writes (write_<label>.log) touched in the last 12 h."""
    jobs, cutoff = [], time.time() - 12 * 3600
    running = cmd(["pgrep", "-af", "download_king.py|write_merge.py"])
    for p in sorted(LOGS.glob("download_*.log")):
        if p.stat().st_mtime < cutoff:
            continue
        lines = p.read_text(errors="replace").strip().splitlines()
        tail = lines[-1] if lines else ""
        digest = p.stem.split("_", 1)[1]
        done = any("checkpoint ready" in l for l in lines[-5:])
        m = re.search(r"total ([\d.]+)/([\d.]+) GB at ([\d.]+) MB/s", " ".join(lines[-20:]))
        alive = f"--digest {digest}" in running
        frac = 1.0 if done else (float(m.group(1)) / float(m.group(2)) if m else None)
        detail = "verified, ready" if done else (f"{m.group(1)}/{m.group(2)} GB at {m.group(3)} MB/s" if m else tail[-90:])
        jobs.append({"name": f"download king {digest}…", "frac": frac, "detail": detail,
                     "status": "done" if done else ("running" if alive else "stopped"),
                     "status_cls": "done" if done else ("busy" if alive else "crit")})
    for p in sorted(LOGS.glob("write_*.log")):
        if p.stat().st_mtime < cutoff:
            continue
        label = p.stem[len("write_"):]
        out = CKPT / label
        lines = p.read_text(errors="replace").strip().splitlines()
        size = sum(f.stat().st_size for f in out.glob("*.safetensors")) if out.exists() else 0
        done = (out / "model.safetensors.index.json").exists() and not re.search(rf"write_merge\.py.*{re.escape(label)}\.json", running)
        alive = bool(re.search(rf"write_merge\.py.*{re.escape(label)}\.json", running))
        failed = not alive and not done
        jobs.append({"name": f"write merge {label}", "frac": 1.0 if done else min(1.0, size / 220.6e9),
                     "detail": f"{size / 1e9:.1f} / 220.6 GB · " + (lines[-1][-80:] if lines else ""),
                     "status": "done" if done else ("running" if alive else "failed"),
                     "status_cls": "done" if done else ("busy" if alive else "crit")})
    return jobs


REPO = Path(__file__).resolve().parents[2]


EVAL_CACHE = Path(os.environ.get("TEUTONIC_EVAL_CACHE", "/data/teutonic/eval_cache"))
OUR_KING_DIGESTS = {"82de662a", "d4e670ff"}  # crowns we hold: t005 = TEXT (uid 88, 07:26 over KT2), t007 = CODE (uid 61, 14:42 over KC2)
KING_NAMES = {"82de662a": "t005 (TEXT king, ours)", "d4e670ff": "t007 (CODE king, ours)", "fa428e5e": "KC2 (CODE king)", "8864c279": "KT2 (TEXT king)", "KT2": "KT2 (TEXT king)", "95ef7a26": "T1 (TEXT king)", "c7ef5487": "KMAIN (MAIN king, uid 107)", "KMATH_ddd9e626_b41": "KMATH (MATH king, uid 137)", "ddd9e626": "KMATH (MATH king, uid 137)", "9b202f1a": "#43", "K43_b41": "#43", "9a39afee": "KTEXT (TEXT king, uid 218)", "main_upper_ema_m1/step-150": "t007 (CODE king, ours)", "7e4aa823": "CK1 (CODE king, ours)",
              "CK3_b41": "#42", "67643e86": "#42", "1f1efb68": "#41", "76d55249": "cypress (TEXT king #5, uid 117)", "TX5_b41": "cypress (TEXT king #5, uid 117)",
              "0c681d78": "KCODE (CODE king, uid 132)", "KCODE_0c681d78_b41": "KCODE (CODE king, uid 132)",
              "414bcb8a": "M45 (MAIN king #45, 5DAL-102)", "M45_b41": "M45 (MAIN king #45, 5DAL-102)"}
CONFIG_COMPETITION = {"0a2840c9": "TEXT", "e8ef9ae1": "MAIN", "7dcbbb5f": "CODE", "abfd40f3": "MATH"}
# our submissions' uids, newest first (TEUTONIC_SUBMISSION_UIDS="56,53"): sn3auto004 = step 150, sn3auto003 = step 75
OUR_UIDS = [int(u) for u in os.environ.get("TEUTONIC_SUBMISSION_UIDS", "88,56,53").split(",") if u.strip()]


def _sample_labels() -> dict[str, dict]:
    out = {}
    for meta in (EVAL_CACHE / "samples").glob("*/meta.json"):
        try:
            m = json.loads(meta.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        out[m["sample_id"]] = {"tag": str(m.get("label", "?")).removeprefix("tag:"), "n": m.get("n"),
                               "competition": CONFIG_COMPETITION.get(m["config_version"][:8], m["config_version"][:8])}
    return out


def evaluations() -> list[dict]:
    """Full validator-size duels (and 2 400-sequence screens) from parity_eval's saved results, one row per
    (competition, king, checkpoint) with the per-sample verdicts (latest full result per sample, else the screen)."""
    samples = _sample_labels()
    reigning = current_kings()
    rows: dict[tuple, dict] = {}
    for f in sorted((EVAL_CACHE / "results").glob("*_duel.json"), key=lambda q: q.stat().st_mtime):
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if data.get("campaign") and not str(data["campaign"]).endswith("full"):  # screens: see merge_campaign()
            continue
        for res in data.get("results", []):
            v, s = res.get("verdict") or {}, samples.get(res.get("sample"), {})
            ch, kg = res.get("challenger") or {}, res.get("king") or {}
            model = ((ch.get("terms") or [{}])[0]).get("model", "")
            run = Path(model).parent.name if "/ckpt/" in model else ""
            king = _king_name(kg)
            if reigning.get(s.get("competition")) and reigning[s["competition"]] not in json.dumps(kg):
                king = king.replace(" king", " king, former").replace("(", "(former: ", 1) if "(" not in king else \
                    re.sub(r"\((.*)\)", r"(former \1)", king)
            key = (s.get("competition", "?"), king, run, ch.get("label") or Path(model).name)
            full = bool(v.get("complete"))
            entry = {"mu": v.get("mu_hat"), "lcb": v.get("lcb"), "accepted": v.get("accepted"), "n": v.get("n_sequences"),
                     "full": full, "t": f.stat().st_mtime, "bar": v.get("delta_threshold"),
                     "text": (v.get("source_scores") or {}).get("text", {}).get("mu_hat"),
                     "docs": (v.get("component_scores") or {}).get("long_documents", {}).get("mu_hat"),
                     "windows": (v.get("component_scores") or {}).get("windows", {}).get("mu_hat")}
            row = rows.setdefault(key, {"competition": key[0], "king": key[1], "run": key[2], "ckpt": key[3], "samples": {}})
            tag = s.get("tag", "?")
            prev = row["samples"].get(tag)
            if prev is None or full or not prev["full"]:  # a full result replaces a screen, never the reverse
                row["samples"][tag] = entry
    out = []
    for row in rows.values():
        full = [e for e in row["samples"].values() if e["full"] and e["mu"] is not None]
        row["mean_full"] = sum(e["mu"] for e in full) / len(full) if full else None
        row["passed"] = sum(1 for e in full if e["accepted"])
        row["n_full"] = len(full)
        row["t"] = max(e["t"] for e in row["samples"].values())
        out.append(row)
    step = lambda r: int(re.sub(r"\D", "", r["ckpt"]) or 0)  # noqa: E731
    return sorted(out, key=lambda r: (r["competition"] != "TEXT", r["competition"], r["king"], r["run"], step(r)))


def _challenger_label(ch: dict) -> str:
    """Recipe label, or <run>/<step> for a single training checkpoint (both runs have a step-25, step-50, ...)."""
    terms = ch.get("terms") or [{}]
    model = Path(terms[0].get("model", "?"))
    if len(terms) == 1 and "/ckpt/" in str(model) and (not ch.get("label") or ch.get("label") == model.name):
        short = model.parent.name.replace("main_upper_ema_", "").replace("text_upper_ema_", "")
        return f"{short}/{model.name}"
    return ch.get("label") or model.name[:12]


def run_screens(run: Path, primary: str = "MAIN") -> list[dict]:
    """Screens of a run's checkpoints vs the reigning kings (each competition vs its own king), newest result per
    checkpoint and competition; the run's start checkpoint (step-0's target) is the reference row. A start that is a
    written merge was screened by the merge campaign under its recipe label (<start>.recipe.json), not as a directory."""
    samples, reigning = _sample_labels(), current_kings()
    start = (run / "step-0").resolve() if (run / "step-0").exists() else None
    start_label = None
    if start is not None:
        try:
            start_label = json.loads(start.with_name(start.name + ".recipe.json").read_text()).get("label")
        except (OSError, json.JSONDecodeError):
            pass
    rows: dict[str, dict] = {}
    for f in sorted((EVAL_CACHE / "results").glob("*_duel.json"), key=lambda q: q.stat().st_mtime):
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for res in data.get("results", []):
            ch = res.get("challenger") or {}
            terms = ch.get("terms") or []
            model = Path(terms[0].get("model", "")) if len(terms) == 1 else None
            if model is not None and model.parent == run:
                key = model.name
            elif start is not None and (model == start or (start_label and ch.get("label") == start_label)):
                key = "start"
            else:
                continue
            comp = samples.get(res.get("sample"), {}).get("competition")
            v = res.get("verdict") or {}
            if comp not in ("MAIN", "MATH", "CODE") or (v.get("n_sequences") or 0) > 6000:
                continue  # screens only (full-sample confirmations are in the duel panel)
            if reigning.get(comp) and reigning[comp] not in json.dumps(res.get("king") or {}):
                continue
            row = rows.setdefault(key, {"ckpt": key, "t": 0})
            row[comp] = {"mu": v.get("mu_hat"), "lcb": v.get("lcb"), "n": v.get("n_sequences")}
            row["t"] = max(row["t"], f.stat().st_mtime)
    step = lambda r: -1 if r["ckpt"] == "start" else int(re.sub(r"\D", "", r["ckpt"]) or 0)  # noqa: E731
    out = sorted(rows.values(), key=step)
    ref = rows.get("start", {}).get(primary, {}).get("mu")
    for r in out:
        mu = r.get(primary, {}).get("mu")
        r["d"] = (mu - ref) if (mu is not None and ref is not None and r["ckpt"] != "start") else None
    return out


# competitions screened for a training run of each kind (train.sh SCREENS; the run's own competition first) and the bars
SCREEN_COMPS = {"main": ["MAIN", "MATH", "CODE"], "math": ["MATH", "MAIN"], "code": ["CODE", "MAIN"]}
BARS = {"MAIN": 0.003, "MATH": 0.002, "CODE": 0.002, "TEXT": 0.002}


def king_names() -> dict[str, str]:
    """Reigning king per competition as its configs/merge/k41/models.json name (MA2, M45, ...), else the digest."""
    try:
        models = json.loads((REPO / "configs/merge/k41/models.json").read_text())
    except (OSError, json.JSONDecodeError):
        models = {}
    out = {}
    for comp, digest in current_kings().items():
        names = [k for k, v in models.items() if digest and digest in str(v)]
        short = [k for k in names if re.fullmatch(r"[A-Z]{1,2}\d+", k)]
        out[comp] = (short or names or [digest or "king"])[0]
    return out


def current_kings() -> dict[str, str]:
    """Digest of the reigning king per competition (MAIN/MATH/CODE/TEXT) from the live dashboards, cached 60 s.
    A vacant MATH crown is contested against the MAIN king."""
    cache = Path("/data/teutonic/www/.kings.json")
    try:
        if cache.exists() and time.time() - cache.stat().st_mtime < 60:
            return json.loads(cache.read_text())
    except (OSError, json.JSONDecodeError):
        pass
    out = {}
    for comp in ("main", "math", "code", "text"):
        url = "https://teutonic.ai/dashboard.json" if comp == "main" else f"https://teutonic.ai/competitions/{comp}/dashboard.json"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "teutonic-miner/1.0"})
            king = json.loads(urllib.request.urlopen(req, timeout=10).read()).get("king") or {}
            out[comp.upper()] = (king.get("king_digest") or "")[:8]
        except Exception:
            out[comp.upper()] = None
    if not out.get("MATH") and out.get("MAIN"):
        out["MATH"] = out["MAIN"]
    if all(out.values()):
        try:
            cache.write_text(json.dumps(out))
        except OSError:
            pass
    elif cache.exists():  # network hiccup: keep the last good answer
        try:
            return {**out, **{k: v for k, v in json.loads(cache.read_text()).items() if not out.get(k)}}
        except (OSError, json.JSONDecodeError):
            pass
    return out


# merge campaigns shown in the merge panel: the running search and the training-checkpoint screens (older rounds hidden;
# their results stay in eval_cache/results). Override: TEUTONIC_DASH_CAMPAIGNS="search3,m1"
SHOW_CAMPAIGNS = set(os.environ.get("TEUTONIC_DASH_CAMPAIGNS", "public,m1,wise_r1s80,wise_r1s80_4800,m4t,ap_").split(","))  # names or prefixes
TOP_PER_COMP = int(os.environ.get("TEUTONIC_DASH_TOP", "8"))  # the merge board keeps the useful rows only (see merge_campaign)


def _shown(campaign) -> bool:
    c = str(campaign)
    return any(c == x or c.startswith(x) for x in SHOW_CAMPAIGNS)
# our crowns are submitted with permuted weights; screens use the un-permuted twin (t007 = m1/step-150)
KING_ALIASES = {"d4e670ff": ("main_upper_ema_m1/step-150",), "82de662a": ("soup_k43_c1_050",)}


def _method_name(ch: dict) -> str:
    """The merge method of a challenger recipe, in the usual names (task arithmetic / soup / TIES / DARE-TIES / ...)."""
    mode, args, terms = ch.get("mode") or "linear", ch.get("mode_args") or {}, ch.get("terms") or []
    if mode == "linear":
        if len(terms) <= 1:
            name = "single model"
        elif any(float(t.get("coef", 0)) < 0 for t in terms):
            name = "task arithmetic"
        else:
            name = "soup (weighted average)"
        if ch.get("overrides"):
            name += " · per layer/module"
        return name
    if mode == "vec_ties":
        if args.get("merge") == "max":
            return "EMR elect-max" + (f" (density {args['density']})" if args.get("density") else "")
        if args.get("della"):
            return f"DELLA-TIES (drop {args.get('drop')}, eps {args['della']})"
        if args.get("drop"):
            return f"DARE-TIES (drop {args['drop']})"
        if args.get("top_drop"):
            return f"breadcrumbs (density {args.get('density')})"
        if args.get("density"):
            return f"TIES (density {args['density']})" + (f" λ{args['lambda']}" if float(args.get("lambda", 1)) != 1 else "")
        return "TIES (sign election)"
    return {"model_stock": "Model Stock", "slerp": "SLERP", "orthogonal": "orthogonal projection", "sign_agree": "sign-agree", "ties": "TIES", "dare": "DARE",
            "dare_ties": "DARE-TIES"}.get(mode, mode)


def _king_name(king_desc: dict) -> str:
    text = json.dumps(king_desc)
    return next((name for key, name in KING_NAMES.items() if key in text), (king_desc.get("label") or "?")[:12])


def _recipe_meta(label: str) -> dict:
    for f in sorted(REPO.glob(f"configs/merge/*/{label}.json"), key=lambda q: -q.stat().st_mtime):
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        return {"desc": d.get("description", ""), "mode": _method_name(d), "group": d.get("group", "")}
    return {"desc": "", "mode": "", "group": ""}


def _merge_board(rows: list[dict], kings: dict, phase: str) -> dict:
    """Four competition cards plus which previous-king training steps are on disk."""
    bars = {"MAIN": 0.003, "MATH": 0.002, "CODE": 0.002, "TEXT": 0.002}
    comps = []
    for comp in ("MAIN", "MATH", "CODE", "TEXT"):
        scored = [r for r in rows if (r.get("cells") or {}).get(comp, {}).get("mu") is not None and r.get("status") == "scored"]
        best = max(scored, key=lambda r: r["cells"][comp]["mu"]) if scored else None
        cell = best["cells"][comp] if best else {}
        info = kings.get(comp) or {}
        comps.append({"comp": comp, "king": info.get("king") or "?", "bar": info.get("bar", bars[comp]),
                      "mu": cell.get("mu"), "lcb": cell.get("lcb"), "method": (best or {}).get("mode") or "",
                      "desc": ((best or {}).get("desc") or "")[:180], "label": (best or {}).get("label") or ""})
    gradients = []
    try:
        public = json.loads((REPO / "configs/merge/public/public_models.json").read_text())
        models = json.loads((REPO / "configs/merge/k41/models.json").read_text())
        reigning = json.loads(Path("/data/teutonic/search/public_kings.json").read_text())
    except (OSError, json.JSONDecodeError):
        public, models, reigning = [], {}, {}
    by_name = {r["name"]: r for r in public}
    for card in comps:
        row = by_name.get(reigning.get(card["comp"].lower(), ""))
        if row:
            card["king"] = f"#{row.get('reign')} {row.get('repo') or row['name']}"
    prefix = {"main": "M", "text": "TX", "code": "CD", "math": "MA"}

    def ingested(name: str) -> bool:
        path = models.get(name)
        return bool(path) and Path(path, "INGESTED").exists()

    def pred(comp: str, reign: int) -> str:
        n = reign - 1
        return f"M{n:02d}" if comp == "main" else f"{prefix[comp]}{n}"

    wanted = [r for r in public if r.get("reign") and (
        (r.get("comp") == "main" and 31 <= int(r["reign"]) <= 45) or r.get("comp") in ("text", "code", "math"))]

    def gkey(r):
        reign = int(r["reign"])
        hot = 0 if r.get("comp") == "main" and 36 <= reign <= 40 else 1
        return (hot, 0 if r.get("comp") == "main" else 1, -reign)

    for r in sorted(wanted, key=gkey):
        reign = int(r["reign"])
        if reign < 1 or r.get("comp") not in prefix:
            continue
        prev = pred(r["comp"], reign)
        have = ingested(r["name"]) and ingested(prev)
        label = f"#{reign}−#{reign - 1}" if r["comp"] == "main" else f"{r['comp'][:1].upper()}#{reign}−#{reign - 1}"
        gradients.append({"label": label, "have": have, "hot": r["comp"] == "main" and 36 <= reign <= 40})
    return {"comps": comps, "gradients": gradients, "phase": phase}


SHARDS = Path("/data/teutonic/eval_cache/shards")


def _line_epoch(ln: str) -> float | None:
    m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", ln)
    if not m:
        return None
    return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()


def _campaign_started() -> float | None:
    for ln in cmd(["pgrep", "-af", "scripts/merge/campaign43.py"]).splitlines():
        if "pgrep" in ln:
            continue
        pid = ln.split(maxsplit=1)[0]
        try:
            return os.stat(f"/proc/{pid}").st_mtime
        except OSError:
            return None
    return None


def _sample_on_disk(total_b: int, shards: int) -> tuple[int, int]:
    """Bytes already stored for this sample, and how many shards are still incomplete.

    A restarted campaign only fetches the missing shards. Counting files touched since the new
    plan line made the bar look like the whole 85 GB was downloading again.
    """
    if shards < 1 or total_b < 1 or not SHARDS.is_dir():
        return 0, shards
    shard_b = total_b // shards
    names = {ent.name for ent in os.scandir(SHARDS)}
    partial_b = 0
    holes = 0
    for name in names:
        if not name.endswith(".partial"):
            continue
        if name[:-len(".partial")] in names:
            continue
        holes += 1
        try:
            partial_b += min(shard_b, (SHARDS / name).stat().st_size)
        except OSError:
            pass
    npy = sum(1 for name in names if name.endswith(".npy"))
    holes += max(0, shards - npy - holes)
    done_shards = max(0, shards - holes)
    return min(total_b, done_shards * shard_b + partial_b), holes


def _campaign_progress(recipes: list[str], tail: list[str]) -> dict:
    """Where campaign43 is: building the sample, loading a model, or scoring its sequences."""
    started = _campaign_started() or 0.0
    fresh = [ln for ln in tail if (_line_epoch(ln) or 0) >= started - 2]
    base = {"stage": "starting", "title": "Preparing the screen", "done": 0, "total": 0, "unit": "",
            "rate": None, "eta_s": None, "batch_done": 0, "batch_total": len(recipes), "detail": "preparing the screen"}
    plan_i = max((i for i, ln in enumerate(fresh) if "long_documents_planned" in ln), default=-1)
    after = fresh[plan_i + 1:] if plan_i >= 0 else fresh
    scoring = any("scoring item " in ln or "total scored " in ln for ln in after)
    if plan_i >= 0 and not scoring:
        plan = fresh[plan_i]
        mb = re.search(r"referenced_shard_bytes': (\d+)", plan)
        ms = re.search(r"unique_shards': (\d+)", plan)
        if not mb or not ms:
            return {**base, "stage": "sample", "title": "Downloading the eval sample",
                    "detail": "downloading the eval sample"}
        total_b = int(mb.group(1))
        shards = int(ms.group(1))
        done_b, holes = _sample_on_disk(total_b, shards)
        rate = None
        stamp = Path("/data/teutonic/search/sample_rate.json")
        now = time.time()
        try:
            prev = json.loads(stamp.read_text())
            dt = now - float(prev["t"])
            gained = done_b - int(prev["done"])
            # a recount of shards already on disk is not a download rate
            if 8 <= dt <= 90 and 0 < gained < 2_000_000_000:
                rate = gained / dt
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
        try:
            stamp.write_text(json.dumps({"t": now, "done": done_b}))
        except OSError:
            pass
        eta = int((total_b - done_b) / rate) if rate and done_b < total_b else None
        if holes and holes < shards:
            title = f"Finishing the eval sample, {holes} shards left"
        else:
            title = "Downloading the eval sample"
        detail = f"{title}: {done_b / 1e9:.1f}/{total_b / 1e9:.1f} GB"
        return {**base, "stage": "sample", "title": title, "done": done_b, "total": total_b,
                "unit": "bytes", "rate": rate, "eta_s": eta, "detail": detail}
    if plan_i < 0 and not scoring:
        n_src = sum(1 for ln in fresh if "pretokenized_source_sampled" in ln)
        if n_src:
            return {**base, "stage": "sample", "title": "Building the eval sample", "done": n_src, "total": 5,
                    "unit": "sources", "detail": f"building the eval sample: {n_src}/5 sources sampled"}
        return base
    item_i = max((i for i, ln in enumerate(fresh) if "scoring item " in ln), default=-1)
    item = re.search(r"scoring item (\d+) \((\S+)\)", fresh[item_i]) if item_i >= 0 else None
    label = item.group(2) if item else ""
    finished = bool(item) and any(f"item {item.group(1)} ({item.group(2)}): scored" in ln for ln in fresh[item_i + 1:])
    batch_done = sum(1 for name in recipes if any(f"({name}): scored" in ln for ln in fresh))
    if item and not finished:
        seq = next((re.search(r"scoring item \d+ \(\S+\) on set \d+: (\d+) sequences", ln) for ln in reversed(fresh[item_i:]) if " sequences" in ln), None)
        scored = next((re.search(r"total scored (\d+)/(\d+) \(([\d.]+) seq/s, eta (\d+)s\)", ln) for ln in reversed(fresh[item_i:]) if "total scored " in ln), None)
        done = int(scored.group(1)) if scored else 0
        total = int(scored.group(2)) if scored else (int(seq.group(1)) if seq else 0)
        rate = float(scored.group(3)) if scored else None
        eta = int(scored.group(4)) if scored else None
        title = f"Scoring {label}" if label in recipes else f"Scoring king {label}"
        detail = f"{title}: {done}/{total} sequences" + (f" at {rate:.0f}/s, eta {eta}s" if rate is not None else "")
        return {**base, "stage": "scoring", "title": title, "done": done, "total": total, "unit": "sequences",
                "rate": rate, "eta_s": eta, "batch_done": batch_done, "detail": detail}
    nxt = next((re.search(r"loading item (\d+) \((\S+)\)", ln) for ln in reversed(fresh) if "loading item " in ln), None)
    name = (nxt.group(2) if nxt else label) or "the next model"
    return {**base, "stage": "loading", "title": f"Loading {name} onto the GPUs", "batch_done": batch_done,
            "detail": f"loading {name} onto the GPUs"}


def merge_campaign() -> dict:
    """Merge campaigns (scripts/merge/campaign43.py): every recipe screened on the four competitions' masked samples
    (first 2 400 sequences of tag:A) vs that competition's king. One row per recipe, one cell per competition, plus the
    recipes the running campaign still has queued and what it is doing right now."""
    samples = _sample_labels()
    rows: dict[str, dict] = {}
    # only verdicts against the REIGNING king of each competition are shown (older ones are not "vs" the king any
    # more and stay out until the recipe is rescored); headers name the reigning kings
    reigning = current_kings()
    bars = {"MAIN": 0.003, "MATH": 0.002, "CODE": 0.002, "TEXT": 0.002}
    kings = {c: {"king": next((n for k, n in KING_NAMES.items() if d and k.startswith(d[:8])), d or "?"), "bar": bars[c]}
             for c, d in reigning.items()}
    for f in sorted((EVAL_CACHE / "results").glob("*_duel.json"), key=lambda q: q.stat().st_mtime):
        try:
            data = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not data.get("campaign") or str(data["campaign"]).endswith("full"):  # full confirmations: evaluations()
            continue
        if not _shown(data["campaign"]):  # user 21:00 JST Oct 11 / Oct 14: older merge rounds are confusing
            continue
        for res in data.get("results", []):
            v, s = res.get("verdict") or {}, samples.get(res.get("sample"), {})
            ch, kg = res.get("challenger") or {}, res.get("king") or {}
            label = _challenger_label(ch)
            comp = s.get("competition", "?")
            kg_text = json.dumps(kg)
            if reigning.get(comp) and reigning[comp] not in kg_text and not any(a in kg_text for a in KING_ALIASES.get(reigning[comp], ())):
                continue  # scored against a former king
            king = _king_name(kg)
            bar = v.get("delta_threshold")
            kings[comp] = {"king": king, "bar": bar}
            row = rows.setdefault(label, {"label": label, "campaign": data["campaign"], "cells": {}, **_recipe_meta(label),
                                          "mode": _method_name(ch) or _recipe_meta(label)["mode"] or "linear"})
            src = v.get("source_scores") or {}
            comps_ = v.get("component_scores") or {}
            row["cells"][comp] = {"mu": v.get("mu_hat"), "lcb": v.get("lcb"), "bar": bar, "pass": v.get("accepted"), "king": king,
                                  "n": v.get("n_sequences"), "src": {k: x.get("mu_hat") for k, x in src.items()},
                                  "docs": comps_.get("long_documents", {}).get("mu_hat"), "windows": comps_.get("windows", {}).get("mu_hat")}
            row["t"] = f.stat().st_mtime
            row["status"] = "scored"
    for row in rows.values():
        margins = [c["mu"] - (c["bar"] or 0) for c in row["cells"].values() if c["mu"] is not None]
        row["best_margin"] = max(margins) if margins else None
        row["marks"] = []
    # verdict marks (user, 03:35): "submit <COMP>" = LCB above the bar against the reigning king of a competition we do
    # not hold (best one starred); where nothing passes, the highest-mu recipe is the "train base" for that competition
    for comp in ("MAIN", "MATH", "CODE", "TEXT"):
        if (reigning.get(comp) or "")[:8] in OUR_KING_DIGESTS:
            continue  # we hold this crown: beating our own king is not a submission
        scored = [r for r in rows.values() if r["cells"].get(comp, {}).get("mu") is not None]
        passing = sorted((r for r in scored if r["cells"][comp].get("pass")), key=lambda r: -r["cells"][comp]["lcb"])
        for i, r in enumerate(passing):
            r["marks"].append(f"submit {comp}" + (" ★" if i == 0 else ""))
        if not passing and scored:
            max(scored, key=lambda r: r["cells"][comp]["mu"])["marks"].append(f"train base {comp}")
    # user (Oct 14): unnecessary / bad recipes off the board. A row stays when it beats its king somewhere by >= 3e-4 (3 SE of a 1 200-seq screen),
    # is among the TOP_PER_COMP by mu of any competition, carries a verdict mark, or is a training checkpoint screen
    # (autopilot ap_* campaign). Everything else stays in eval_cache/results, just not on the page.
    keep: set[str] = set()
    for comp in ("MAIN", "MATH", "CODE", "TEXT"):
        scored = [r for r in rows.values() if r["cells"].get(comp, {}).get("mu") is not None]
        for r in sorted(scored, key=lambda r: -r["cells"][comp]["mu"])[:TOP_PER_COMP]:
            keep.add(r["label"])
    for r in rows.values():
        if r["marks"] or str(r.get("campaign", "")).startswith("ap_") or any((c.get("mu") or -1) >= 3e-4 for c in r["cells"].values()):
            keep.add(r["label"])
    hidden = len(rows) - len(keep)
    rows = {k: r for k, r in rows.items() if k in keep}
    # the running campaign: its recipe queue (command line) and its current phase (log tail)
    running, phase, progress = [], "", None
    for line in cmd(["pgrep", "-af", "scripts/merge/campaign"]).splitlines():
        running += [Path(x).stem for x in re.findall(r"--recipe\s+(\S+)", line)]
    if running:
        logs = sorted(LOGS.glob("pm2_campaign*.log"), key=lambda q: q.stat().st_mtime)
        tails = [q.read_text(errors="replace")[-30000:].splitlines() for q in logs[-3:]]
        tail = next((t for t in reversed(tails) if any(" INFO " in ln for ln in t)), [])  # skip launchers still waiting
        last = next((ln for ln in reversed(tail) if re.search(r"total scored|loaded in|INFO parity_eval: scoring|scoring item|campaign43: king", ln)), "")
        m = re.search(r"total scored (\d+)/(\d+) \(([\d.]+) seq/s, eta (\d+)s\)", last)
        if any("mppool" in ln for ln in tail):  # ping-pong pool: one set scores, the other preloads the next recipe
            cur = next((re.search(r"scoring item (\d+) \((\S+)\)", ln) for ln in reversed(tail) if "scoring item" in ln), None)
            done_ = cur and any(f"item {cur.group(1)} ({cur.group(2)}): scored" in ln for ln in tail)
            nxt = next((re.search(r"loading item (\d+) \((\S+)\)", ln) for ln in reversed(tail) if "loading item" in ln), None)
            ready = nxt and any(f"item {nxt.group(1)} loaded in" in ln for ln in tail)
            phase = (f"scoring {cur.group(2)}" + (f": {m.group(1)}/{m.group(2)} sequences at {float(m.group(3)):.0f}/s, eta {int(m.group(4))} s"
                                                    if m and "total scored" in last else "") if cur and not done_ else "waiting for a load")
            if nxt and (not cur or int(nxt.group(1)) > int(cur.group(1))):
                phase += f" · next {nxt.group(2)} {'loaded, waiting' if ready else 'preloading on the other replica set'}"
        else:
            what = next((re.sub(r".*scoring (\S+) on .*", r"\1", ln) for ln in reversed(tail) if "INFO parity_eval: scoring" in ln), "")
            phase = (f"scoring {what}: {m.group(1)}/{m.group(2)} sequences, eta {int(m.group(4))} s" if m else
                     f"{what}: replicas loaded, scoring" if "loaded in" in last else f"loading {what} (merge + upload to the GPUs)" if what else "starting")
        pending = [x for x in running if x not in rows]
        current = pending[0] if pending else None
        progress = _campaign_progress(running, tail)
        if progress["stage"] == "sample":
            phase = progress["detail"]
        elif current and current not in phase:  # the log tail belongs to an older campaign (searches log elsewhere)
            phase = f"scoring {current}: {(_recipe_meta(current).get('desc') or '')[:90]}"
        if current:  # only the recipe being scored gets a row; the queue is a count (user: the panel was too long)
            rows[current] = {"label": current, "cells": {}, **_recipe_meta(current), "status": "scoring", "best_margin": None, "t": None, "marks": []}
        queued_n = max(0, len(pending) - 1)
    order = {"scoring": 0, "scored": 1, "queued": 2}
    out = sorted(rows.values(), key=lambda r: (not any(m.startswith("submit") for m in r["marks"]), not r["marks"],
                                               order.get(r["status"], 3), -(r["best_margin"] if r["best_margin"] is not None else -1)))
    board = _merge_board(out, kings, phase)
    return {"rows": out, "kings": kings, "phase": phase, "queued": queued_n if running else 0, "summary": public_summary(),
            "progress": progress, "board": board, "hidden": hidden}


def public_summary() -> str:
    """One line on the public-model campaign (scripts/merge/public_merge.py state): models probed, kept, best per competition."""
    f = Path("/data/teutonic/search/public.json")
    try:
        st = json.loads(f.read_text())
        total = len(json.loads((REPO / "configs/merge/public/public_models.json").read_text()))
    except (OSError, json.JSONDecodeError):
        return ""
    pr = st.get("probes", {})
    family = {"M44", "M43", "M42", "M41", "M40", "TX1", "TX2", "TX3", "TX4", "TX5", "CD1", "CD2", "CD3", "CD4", "MA1"}
    public = [n for n in pr if not n.startswith("C") or n in family]
    done = [n for n in public if pr[n].get("gain")]
    old_done = [n for n in done if n not in family]
    deleted = [n for n in old_done if pr[n].get("deleted")]
    best = st.get("best", {})
    left = total - len(done)
    eta = time.time() + left * 2 * 95 + 1800  # two probes per model at ~95 s + one search round
    parts = [f"{c.upper()} {b['mu']:+.6f} (bar {b['bar']})" for c, b in best.items() if b.get("mu") is not None]
    return (f"public-model campaign: {len(done)}/{total} public kings probed toward/away from anchor {st.get('anchor')} "
            f"({len(done) - len(old_done)} near-family kings, {len(old_done)} older kings: {len(old_done) - len(deleted)} kept on /data as merge directions, "
            f"{len(deleted)} moved to the 13 TB cold volume after probing); {len(st.get('pool', {}))} candidates scored; "
            f"best so far: {' · '.join(parts) or '–'}; all models probed ≈ {dt.datetime.fromtimestamp(eta, JST):%m-%d %H:%M} JST")


def eval_jobs() -> list[dict]:
    """pm2 eval_* / sample_* jobs with their latest progress line (scoring rate and ETA, or what they wait for)."""
    jobs = []
    for log in sorted(LOGS.glob("pm2_eval_*.log")) + sorted(LOGS.glob("pm2_sample_*.log")):
        name = log.stem.removeprefix("pm2_")
        if time.time() - log.stat().st_mtime > 6 * 3600:
            continue
        tail = log.read_text(errors="replace")[-20000:].splitlines()
        progress = next((ln for ln in reversed(tail) if "total scored" in ln or "waiting for" in ln or "sampler {" in ln), "")
        m = re.search(r"total scored (\d+)/(\d+) \(([\d.]+) seq/s, eta (\d+)s\)", progress)
        scoring = next((ln for ln in reversed(tail) if "INFO parity_eval: scoring" in ln), "")
        # running = its launcher script is alive, or the log moved in the last 90 s (a job may run several duels in a row)
        alive = bool(cmd(["pgrep", "-f", f"/data/teutonic/logs/{name}.sh"])) or time.time() - log.stat().st_mtime < 90
        finished = any("training resumed" in ln for ln in tail[-3:]) or (name.startswith("sample_") and "windows +" in "".join(tail[-3:]))
        detail = (f"{m.group(1)}/{m.group(2)} at {float(m.group(3)):.0f}/s, eta {int(m.group(4)) // 60} min" if m
                  else re.sub(r"^\S+: ", "", progress)[-140:])
        what = re.sub(r".*scoring (\S+) on (\S+): (\d+) sequences.*", r"\1 on \2 (\3)", scoring)[-80:] if scoring else ""
        jobs.append({"name": name, "frac": int(m.group(1)) / int(m.group(2)) if m else None, "detail": f"{what} · {detail}" if what else detail,
                     "status": "running" if (alive and not finished) else "done", "t": log.stat().st_mtime})
    return sorted(jobs, key=lambda j: -j["t"])[:6]


def validator_duel() -> dict:
    """Our submission on the live TEXT dashboard (queue / running stage + provisional mu / verdicts), cached 60 s."""
    cache = Path("/data/teutonic/www/.validator_text.json")
    d = None
    if cache.exists() and time.time() - cache.stat().st_mtime < 60:
        try:
            d = json.loads(cache.read_text())
        except (OSError, json.JSONDecodeError):
            d = None
    if d is None:
        try:
            req = urllib.request.Request("https://teutonic.ai/competitions/text/dashboard.json", headers={"User-Agent": "teutonic-miner/1.0"})
            d = json.loads(urllib.request.urlopen(req, timeout=10).read())
            cache.write_text(json.dumps({k: d.get(k) for k in ("king", "queue", "current_eval", "history")}))
        except Exception:
            d = json.loads(cache.read_text()) if cache.exists() else {}
    ce = d.get("current_eval") or {}
    mine = ce if ce.get("uid") in OUR_UIDS else {}
    return {
        "uid": ", ".join(map(str, OUR_UIDS)), "king": (d.get("king") or {}).get("model_repo"),
        "queue": [f"uid {x.get('uid')}: {x.get('state')}" for x in (d.get("queue") or []) if x.get("uid") in OUR_UIDS],
        "running": {k: mine.get(k) for k in ("challenge_id", "stage", "percent", "progress", "total", "provisional_mu_hat",
                                              "provisional_lcb", "elapsed_seconds", "started_at")} if mine else None,
        "history": [{k: h.get(k) for k in ("uid", "challenge_id", "verdict", "accepted", "mu_hat", "lcb", "error_code", "timestamp")}
                    for h in (d.get("history") or []) if h.get("uid") in OUR_UIDS][-4:],
    }


# ----------------------------------------------------------------------------- autopilot (scripts/train/autopilot.py)
# one controller per competition writes <dir>/<comp>/status.json (+ state.json with the full history) every ~30 s and
# lists staged submission candidates in autopilot_ready.json. Override for tests: TEUTONIC_AUTOPILOT_DIR / _READY
AUTOPILOT = Path(os.environ.get("TEUTONIC_AUTOPILOT_DIR", "/data/teutonic/autopilot"))
AUTOPILOT_READY = Path(os.environ.get("TEUTONIC_AUTOPILOT_READY", "/data/teutonic/search/autopilot_ready.json"))
AP_COMPS = ("main", "math", "code", "text")


def _jread(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _stage_state(log) -> dict:
    """How far a staging job got (autopilot.py STAGE_SH: permute + verify, STAGED, HF backup, STAGE DONE; set -e)."""
    try:
        text = Path(log).read_text(errors="replace")[-20000:]
    except (OSError, TypeError):
        return {"state": "no stage log yet", "cls": "pending", "line": ""}
    lines = [ln for ln in text.splitlines() if ln.strip()]
    last = lines[-1][-180:] if lines else ""
    if "STAGE DONE" in text:
        failed = "HF backup failed" in text
        return {"state": "staged · HF backup failed" if failed else "staged + HF backup done", "cls": "busy" if failed else "done", "line": last}
    if "NOT READY" in text or "Traceback" in text or re.search(r"^FAIL ", text, re.M):
        return {"state": "staging FAILED: read the log", "cls": "crit", "line": last}
    if "STAGED" in text:
        return {"state": "staged · HF backup running", "cls": "done", "line": last}
    return {"state": "staging: hidden permutation + verify", "cls": "busy", "line": last}


def _ap_marks(st: dict, train: list[dict]) -> list[dict]:
    """Chart markers for the run on screen when it is an autopilot run: every step where the trainer's logged lr_scale
    changes (reason from the controller's LR history / anneal results; after a rollback the abandoned branch's rows, and
    so its changes, are gone from train_log.jsonl), plus anneals (event time -> the step logged at that time)."""
    lr = st.get("lr") or {}
    hist = [h for h in (lr.get("history") or []) if isinstance(h, dict) and _num(h.get("to")) is not None]
    events = [e for e in (st.get("events") or []) if isinstance(e, dict) and _num(e.get("t")) is not None]
    timed = [(r["t"], r["step"]) for r in train if _num(r.get("t")) is not None and "step" in r]
    t_first = timed[0][0] if timed else None

    def step_at(t: float) -> int | None:
        if t_first is None or t < t_first - 600:
            return None  # before this run (an earlier segment)
        return max((s for tt, s in timed if tt <= t), default=timed[0][1])

    marks, used, prev = [], set(), None
    for r in train:
        sc = _num(r.get("lr_scale"))
        if sc is None:
            continue
        if prev is not None and abs(sc - prev) > 1e-9:
            cand = [i for i, h in enumerate(hist) if i not in used and abs(float(h["to"]) - sc) <= 1e-3 * max(sc, 1e-6)]
            i = min(cand, key=lambda i: abs(int(hist[i].get("step") or 0) - r["step"]), default=None)
            res = next((e for e in events if e.get("kind") == "anneal_result" and _num(e.get("next_scale")) is not None
                        and abs(float(e["next_scale"]) - sc) <= 2e-3 * max(sc, 1e-6) and step_at(e["t"]) is not None
                        and abs(step_at(e["t"]) - r["step"]) <= 5), None)
            if i is not None:
                used.add(i)
                h = hist[i]
                act = h.get("act") or "scale"
                why = h.get("why") or ""
                if act == "rollback":
                    why = f"decided at step {h.get('step')}, back to {h.get('state') or 'the best state'}: {why}"
            elif res is not None:
                act, why = "restore", f"anneal screened ({res.get('ckpt')}), back to {res.get('restore')} at a higher scale"
            else:
                act, why = ("raise" if sc > prev else "lower"), "lr_scale changed (no controller decision on record)"
            marks.append({"step": r["step"], "kind": act if act in ("raise", "lower", "rollback", "restore") else "scale",
                          "label": f"{act} ×{prev:.2f}→×{sc:.2f}", "why": why})
        prev = sc
    for e in events:
        if e.get("kind") != "anneal" or (s := step_at(e["t"])) is None:
            continue
        if not any(m["kind"] == "anneal" and abs(m["step"] - s) <= 5 for m in marks):
            marks.append({"step": s, "kind": "anneal", "label": f"anneal {e.get('steps', '?')} steps", "why": str(e.get("why") or "")})
    if lr.get("anneal") and _num(lr.get("anneal_from")) is not None:
        s = int(lr["anneal_from"])
        if not any(m["kind"] == "anneal" and abs(m["step"] - s) <= 5 for m in marks):
            marks.append({"step": s, "kind": "anneal", "label": f"anneal {lr['anneal']} steps (running)", "why": str((lr.get("last") or {}).get("why") or "")})
    return sorted(marks, key=lambda m: m["step"])


def _ap_card(d: Path, st: dict, run_arg: Path, bars: dict) -> dict:
    comp = str(st.get("comp") or d.name).lower()
    run = st.get("run")
    rd = run_arg if run and run_arg.name == run else (CKPT / run if run else None)
    a = trainer_args(run) if run else {}
    lr = st.get("lr") or {}
    rows = jsonl(rd / "train_log.jsonl") if rd else []
    tr = next((r for r in reversed(rows) if "train" in r), None)
    vl = next((r for r in reversed(rows) if "val" in r), None)
    base, scale = _num(a.get("lr")), _num(lr.get("scale"))
    ctl = _jread(rd / "CONTROL.json") if rd else None
    ack = _jread(rd / "CONTROL.ack.json") if rd else None
    kings = st.get("kings") or {}
    ls = st.get("last_screen") or {}
    res, kn = ls.get("res") or {}, ls.get("kings") or {}
    srows = []
    for c in AP_COMPS:
        v = res.get(c) or {}
        mu, bar = _num(v.get("mu")), bars[c]
        srows.append({"comp": c.upper(), "mu": mu, "lcb": _num(v.get("lcb")), "n": v.get("n"), "king": kn.get(c) or v.get("king"),
                      "bar": bar, "gap": (mu - bar) if mu is not None else None, "sources": v.get("sources") or {}})
    best = []
    for c in AP_COMPS:
        b = (st.get("best") or {}).get(c)
        if not isinstance(b, dict):
            continue
        live, kd = str((kings.get(c) or {}).get("digest") or ""), str(b.get("king_digest") or "")
        best.append({"comp": c.upper(), "mu": _num(b.get("mu")), "lcb": _num(b.get("lcb")), "run": b.get("run"), "ckpt": b.get("ckpt"),
                     "path": b.get("path"), "king": b.get("king"), "t": b.get("t"), "bar": bars[c],
                     "former": bool(live and kd and live[:12] != kd[:12])})
    events = [e for e in (st.get("events") or []) if isinstance(e, dict)][-8:][::-1]
    return {
        "comp": comp.upper(), "t": _num(st.get("t")) or (d / "status.json").stat().st_mtime, "phase": st.get("phase") or "starting",
        "run": run, "start": st.get("start"), "run_dir": str(rd) if rd else None,
        "lr": {"scale": scale, "since": lr.get("since"), "ceiling": _num(lr.get("ceiling")), "best_val": _num(lr.get("best_val")),
               "best_step": lr.get("best_step"), "anneal": lr.get("anneal") or 0, "anneal_from": lr.get("anneal_from"),
               "last": lr.get("last"), "history": (lr.get("history") or [])[-3:], "n_changes": len(lr.get("history") or []),
               "base": base, "dense": _num(a.get("lr-dense-frac")) or 1.0, "eff": base * scale if base is not None and scale is not None else None,
               "now": _num(tr.get("lr")) if tr else None, "now_scale": _num(tr.get("lr_scale")) if tr else None, "now_step": tr.get("step") if tr else None,
               "val": _num(vl.get("val")) if vl else None, "val_step": vl.get("step") if vl else None,
               "control": {k: ctl.get(k) for k in ("id", "lr_scale", "anneal")} if isinstance(ctl, dict) else None,
               "ack": {k: ack.get(k) for k in ("id", "step", "lr", "lr_scale", "anneal", "note", "t")} if isinstance(ack, dict) else None},
        "screen": {"run": ls.get("run"), "ckpt": ls.get("ckpt"), "step": ls.get("step"), "t": ls.get("t"), "branch": ls.get("branch") or 0,
                   "annealed": bool(ls.get("annealed")), "skipped": ls.get("skipped"), "failed": ls.get("failed"), "rows": srows} if ls else None,
        "best": best, "events": events,
        "kings": {c.upper(): {"reign": k.get("reign"), "repo": k.get("repo"), "digest": str(k.get("digest") or "")[:8]}
                  for c, k in kings.items() if isinstance(k, dict)},
    }


def autopilot(run: Path, train: list[dict]) -> dict:
    """One card per competition's autopilot status.json, the staged submission candidates, and chart markers for the
    run on screen. A missing / empty directory gives no cards (the panel stays hidden); a card that does not parse shows
    its error instead of breaking the render."""
    try:
        dirs = sorted((p for p in AUTOPILOT.iterdir() if p.is_dir()),
                      key=lambda p: (AP_COMPS.index(p.name) if p.name in AP_COMPS else 9, p.name))
    except OSError:
        dirs = []
    statuses, cards, marks = [], [], []
    for d in dirs:
        st = _jread(d / "status.json")
        if isinstance(st, dict):
            statuses.append((d, st))
        elif (d / "status.json").exists():  # written atomically by the controller: unparseable = a real problem
            cards.append({"comp": d.name.upper(), "t": (d / "status.json").stat().st_mtime, "error": "status.json does not parse"})
    # bars: the published defaults, else the newest bar the controllers used for a confirmation / staging
    bars = {c: BARS[c.upper()] for c in AP_COMPS}
    pool = [r for r in (_jread(AUTOPILOT_READY, []) or []) if isinstance(r, dict)]
    for _, st in statuses:
        pool += [r for r in (st.get("ready") or []) if isinstance(r, dict)]
    for r in sorted(pool, key=lambda r: _num(r.get("t")) or 0):
        if str(r.get("comp") or "").lower() in bars and _num(r.get("bar")) is not None:
            bars[str(r["comp"]).lower()] = float(r["bar"])
    for d, st in statuses:
        try:
            cards.append(_ap_card(d, st, run, bars))
            if st.get("run") == run.name:  # the run on screen: markers from the full history (state.json), else status
                full = _jread(d / "state.json")
                marks = _ap_marks(full if isinstance(full, dict) and full.get("lr") else st, train)
        except Exception as exc:  # noqa: BLE001 - a new controller's schema must never take the dashboard down
            cards.append({"comp": d.name.upper(), "t": _num(st.get("t")), "error": f"{type(exc).__name__}: {exc}"[:300]})
    cards.sort(key=lambda c: AP_COMPS.index(c["comp"].lower()) if c["comp"].lower() in AP_COMPS else 9)
    ready, seen = [], set()
    for r in sorted(pool, key=lambda r: -(_num(r.get("t")) or 0)):
        key = (str(r.get("comp")).lower(), r.get("staged_dir") or r.get("path"))
        if key in seen:
            continue
        seen.add(key)
        comp = str(r.get("comp") or "?").upper()
        kd = str(r.get("king_digest") or "")
        cur = next((c["kings"][comp]["digest"] for c in sorted(cards, key=lambda c: -(c.get("t") or 0))
                    if ((c.get("kings") or {}).get(comp) or {}).get("digest")), "")  # the freshest controller's view
        full = {k: {"mu": _num(v.get("mu")), "lcb": _num(v.get("lcb")), "n": v.get("n")} for k, v in (r.get("full") or {}).items() if isinstance(v, dict)}
        ready.append({"comp": comp, "path": r.get("path"), "staged_dir": r.get("staged_dir"), "stage_log": r.get("stage_log"),
                      "king": r.get("king"), "bar": _num(r.get("bar")), "full": full, "t": _num(r.get("t")),
                      "staged_exists": bool(r.get("staged_dir")) and Path(str(r["staged_dir"])).exists(),
                      "former": bool(kd and cur and kd[:8] != cur[:8]), "stage": _stage_state(r.get("stage_log"))})
    now = time.time()
    return {"dir": str(AUTOPILOT), "ready_file": str(AUTOPILOT_READY), "cards": cards, "ready": ready, "marks": marks,
            "fresh": any(c.get("t") and now - c["t"] < 3600 for c in cards)}


# ----------------------------------------------------------------------------- the current run, live
def screen_interval(run: Path, a: dict) -> int:
    """Steps between king screens: the plan's screen_every (train.sh SCREEN_EVERY) for this run's lineage, else every save.
    Autopilot runs (<comp>_ap_<MMDDHHMM>): the controller's --screen-every (autopilot.py BASE: 40)."""
    save_every = int(a.get("save-every", 100))
    if m := re.match(r"(main|math|code|text)_ap_", run.name):
        for line in cmd(["pgrep", "-af", "scripts/train/autopilot.py"]).splitlines():
            if re.search(rf"--comp[ =]{m.group(1)}\b", line) and (s := re.search(r"--screen-every[ =](\d+)", line)):
                return int(s.group(1))
        return 40
    try:
        plan = json.loads((CKPT / "current_plan.json").read_text())
    except (OSError, json.JSONDecodeError):
        return save_every
    return int(plan.get("screen_every") or save_every) if plan.get("base") and run.name.startswith(plan["base"]) else save_every


def run_timeline(run: Path, a: dict) -> dict:
    """Merge per-step jsonl rows (with "t") and the pm2 log into one timeline; derive phase, rates and next events."""
    name = run.name
    steps, save_every, eval_every = int(a.get("steps", 800)), int(a.get("save-every", 100)), int(a.get("eval-every", 100))
    screen_every = screen_interval(run, a)  # the trainer is paused for a king screen after these saves only
    rows = jsonl(run / "train_log.jsonl")
    lines = log_lines(name)
    ev: list[tuple[float, str, int | None, str]] = []  # (t, kind, step, text)
    timed = any("t" in r for r in rows if "train" in r)
    for t, msg in lines:
        if m := STEP_LINE.match(msg):
            if not timed:
                ev.append((t, "step", int(m.group(1)), msg))
        elif m := VAL_LINE.match(msg):
            ev.append((t, "val", int(m.group(1)), msg))
        elif msg.startswith("saved "):
            m = re.search(r"/(step-(\d+)|final) ", msg)
            ev.append((t, "saved", int(m.group(2)) if m and m.group(2) else steps, msg))
        elif msg.startswith("paused"):
            ev.append((t, "paused", None, msg))
        elif msg.startswith("resumed"):
            ev.append((t, "resumed", None, msg))
        elif msg.startswith("model built"):
            ev.append((t, "built", None, msg))
        elif msg.startswith("FSDP ready"):
            ev.append((t, "fsdp", None, msg))
        elif "Traceback" in msg or "Error" in msg:
            ev.append((t, "error", None, msg))
    if timed:
        ev += [(r["t"], "step", r["step"], "") for r in rows if "train" in r and "t" in r]
    ev.sort(key=lambda e: e[0])
    started = lines[0][0] if lines else None
    last_line = next(((t, m) for t, m in reversed(lines) if m.strip()), (None, None))

    # rates: pure step time between consecutive step events with nothing else in between
    stepv = [(t, s) for t, k, s, _ in ev if k == "step"]
    rates = []
    for (t0, s0), (t1, s1) in zip(stepv, stepv[1:]):
        if s1 > s0 and not any(t0 < t < t1 and k != "step" for t, k, _, _ in ev):
            rates.append((t1 - t0) / (s1 - s0))
    step_rows = [r for r in rows if "train" in r and r.get("step_s")]
    # only steps since the trainer's last (re)start (a gap of > 10 min between rows), minus that start's first step
    # (kernel compile): after a resume with a faster trainer the median of the last 10 rows showed the old speed
    cut = max((i for i in range(1, len(step_rows)) if step_rows[i].get("t", 0) - step_rows[i - 1].get("t", 0) > 600), default=0)
    recent = step_rows[cut + 1:] if len(step_rows) - cut > 1 else step_rows[cut:]
    if recent:
        step_s = statistics.median(r["step_s"] for r in recent[-10:])
    elif rates:
        step_s = statistics.median(rates[-5:])
    else:
        step_s = None
    # measured durations of val, save, scoring pause (defaults until the run has its own)
    by_step = {s: t for t, k, s, _ in ev if k == "step"}
    val_d = [t - by_step[s] for t, k, s, _ in ev if k == "val" and s in by_step and s > 0]
    save_d = []
    for t, k, s, _ in ev:
        if k == "saved":
            prev = [tt for tt, kk, ss, _ in ev if ss == s and kk in ("step", "val") and tt <= t]
            if prev:
                save_d.append(t - max(prev))
    pause_d, t_p = [], None
    for t, k, _, _ in ev:
        if k == "paused":
            t_p = t
        elif k == "resumed" and t_p:
            pause_d.append(t - t_p)
            t_p = None
    val_s = statistics.median(val_d) if val_d else 15.0
    save_s = statistics.median(save_d) if save_d else 240.0
    pause_s = (statistics.median(pause_d) if pause_d else 200.0) + 80.0  # + scorer's poll/load lag before the pause

    last_step = max((s for _, k, s, _ in ev if k == "step"), default=0)
    last = ev[-1] if ev else None
    now = time.time()
    phase, detail, progress = "loading model", "", float(last_step)
    if last:
        t, k, s, _ = last
        if k == "built":
            phase = "sharding the model (FSDP init)"
        elif k == "fsdp":
            phase = "step-0 validation"
        elif k == "error":
            phase = "error in the trainer log"
        elif k == "paused":
            phase = "paused: GPUs lent to the scorer"
        elif k == "saved":
            phase = f"saved step-{s}; scorer pauses training to score it" if s % screen_every == 0 and s != steps else "saved"
        elif k == "val" and s and (s % save_every == 0 or s == steps):
            phase = f"saving step-{s} (≈{save_s / 60:.0f} min)"
        elif k == "step" and s and (s % eval_every == 0 or s == steps):
            phase = f"validating after step {s}"
        elif k == "step" and s and s % save_every == 0:
            phase = f"saving step-{s} (≈{save_s / 60:.0f} min)"
        else:  # training (after a step, val, resume or step-0 val)
            cur = last_step + 1
            if step_s:
                into = now - t
                frac = min(0.99, into / step_s)
                progress = last_step + frac
                if timed:
                    phase, detail = f"training step {cur}", f"{frac * 100:.0f}% through this step (≈{max(0, step_s - into):.0f} s left), {step_s:.0f} s per step"
                else:  # pm2 step lines come every --log-every steps: estimate where it is between them
                    est = last_step + int(into // step_s)
                    progress = last_step + into / step_s
                    phase, detail = f"training, ≈ step {est + 1}", f"last logged step {last_step} at {dt.datetime.fromtimestamp(t, JST):%H:%M:%S} JST; {step_s:.0f} s per step"
            else:
                phase = f"training step {cur}"
    # next events and the finish time: remaining steps + vals + saves + scoring pauses ahead
    nxt, finish = [], None
    if step_s and last and phase != "error in the trainer log":
        t_last, k, s_last, _ = last
        pending = 0.0  # work still owed for the last event's step before the next training step starts
        if k == "step" and s_last and (s_last % eval_every == 0 or s_last == steps):
            pending += val_s
        if k in ("step", "val") and s_last and s_last % save_every == 0:
            pending += save_s + (pause_s if s_last != steps and s_last % screen_every == 0 else 0)
        elif k == "saved" and s_last != steps and s_last % screen_every == 0:
            pending += pause_s
        elif k == "paused":
            pending += max(0.0, pause_s - 80.0)
        t, nv, ns, nsc = max(now - (progress - last_step) * step_s, t_last + pending), None, None, None
        for s in range(last_step + 1, steps + 1):
            t += step_s
            if s % eval_every == 0 or s == steps:
                t += val_s
                if nv is None:
                    nv = ("val at step %d" % s, t)
            if s % save_every == 0 or s == steps:
                t += save_s
                if ns is None:
                    ns = (f"step-{s} saved", t, s)
                if s != steps and s % screen_every == 0:
                    if nsc is None:
                        nsc = (s, t + pause_s)
                    t += pause_s
        finish = t
        if nv:
            nxt.append({"what": nv[0], "t": nv[1]})
        if ns:
            nxt.append({"what": ns[0], "t": ns[1]})
        if nsc:
            nxt.append({"what": f"step-{nsc[0]} king screen", "t": nsc[1]})
        nxt.append({"what": "run finishes", "t": finish})
    tps = (int(a.get("batch", 2)) * int(a.get("accum", 16)) * int(a.get("_world", 8)) * _seq_len(a) / step_s) if step_s else None
    return {"phase": phase, "detail": detail, "progress_steps": progress, "started": started, "step_s": step_s, "tok_s": tps,
            "finish": finish, "next": nxt, "last_line": (last_line[1] or "")[:220], "last_line_t": last_line[0]}


def train_points(run: Path, a: dict) -> tuple[list[dict], list[dict], int]:
    """Train rows (with per-step tok/s: from the row when logged, else from pm2 timestamps) and val rows."""
    rows = jsonl(run / "train_log.jsonl")
    train = [r for r in rows if "train" in r]
    val = [r for r in rows if "val" in r]
    timed = any("tok_s_step" in r for r in train)
    tok_step = int(a.get("batch", 2)) * int(a.get("accum", 16)) * int(a.get("_world", 8)) * _seq_len(a) if a else None
    if timed:
        for r in train:
            r["tps"] = r.get("tok_s_step")
        every = 1
    else:  # older trainer: rows every --log-every steps; tok/s between consecutive pm2 step lines without val/save/pause between
        lines = log_lines(run.name)
        marks = [(t, int(m.group(1))) for t, msg in lines if (m := STEP_LINE.match(msg))]
        other = [t for t, msg in lines if VAL_LINE.match(msg) or msg.startswith(("saved", "paused", "resumed"))]
        rate = {}
        for (t0, s0), (t1, s1) in zip(marks, marks[1:]):
            if s1 > s0 and tok_step and not any(t0 < t < t1 for t in other):
                rate[s1] = tok_step * (s1 - s0) / (t1 - t0)
        for r in train:
            r["tps"] = rate.get(r["step"])
        every = int(a.get("log-every", 10))
    return train, val, every


def all_runs(status: dict[str, str]) -> list[dict]:
    out = []
    for d in CKPT.iterdir():
        if not d.is_dir() or not (d / "scores.json").exists():
            continue
        try:
            scores = json.loads((d / "scores.json").read_text())
        except ValueError:
            continue
        a = trainer_args(d.name)
        rows = jsonl(d / "train_log.jsonl")
        vals = [r for r in rows if "val" in r]
        lines = log_lines(d.name)
        started = lines[0][0] if lines else (d / "scores.json").stat().st_mtime
        s0 = next((s for s in scores if s.get("ckpt") == "step-0" and s.get("text_mu") is not None), None)
        done = [s for s in scores if s.get("ckpt") != "step-0" and s.get("status") == "done" and s.get("text_mu") is not None]
        best = max(done, key=lambda s: s["text_mu"], default=None)
        st = status.get(d.name)
        if (d / "final").exists() or (d / "CONFIRMED.json").exists():
            label, cls = "finished", "done"
        elif st == "online":
            label, cls = ("paused (scoring)", "busy") if (d / "PAUSED").exists() else ("training", "done")
        else:
            label, cls = "stopped", "idle"
        note = s0.get("note", "") if s0 else ""
        out.append({"name": d.name, "started": started, "recipe": describe(a)[0] if a else (f"merge base: {note}" if note else "–"),
                    "steps": int(a["steps"]) if "steps" in a else None,
                    "last_step": max((r["step"] for r in rows if "train" in r), default=None),
                    "val_delta": (vals[-1]["val"] - vals[0]["val"]) if len(vals) > 1 else None,
                    "val_step": vals[-1]["step"] if len(vals) > 1 else None,
                    "scored": ", ".join(f"{s['ckpt']} {s['text_mu']:+.5f}" for s in done) or None,
                    "start_mu": s0["text_mu"] if s0 else None,
                    "best_delta": (best["text_mu"] - s0["text_mu"]) if best and s0 else None,
                    "best_ckpt": best["ckpt"] if best else None, "status": label, "status_cls": cls})
    return sorted(out, key=lambda r: r["started"] or 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path, help="run dir with train_log.jsonl (and optional scores.json)")
    ap.add_argument("--out", required=True, type=Path, help="index.html; data.json is written next to it")
    ap.add_argument("--name", default="Teutonic TEXT run")
    ap.add_argument("--steps", type=int, default=None, help="default: --steps of the launcher script")
    ap.add_argument("--target", type=float, default=0.0026)
    ap.add_argument("--recipe", default=None, help="default: built from the launcher script's flags")
    ap.add_argument("--tokens-per-step", type=float, default=None, help="default: batch × accum × GPUs × 2048")
    ap.add_argument("--refresh", type=int, default=0, help="unused (the page polls data.json); kept for old callers")
    args = ap.parse_args()

    run = args.run
    a = trainer_args(run.name)
    recipe, schedule = describe(a)
    steps = args.steps or int(a.get("steps", 800))
    tok_step = args.tokens_per_step or (int(a.get("batch", 2)) * int(a.get("accum", 16)) * int(a.get("_world", 8)) * _seq_len(a))
    train, val, every = train_points(run, a)
    scores = json.loads((run / "scores.json").read_text()) if (run / "scores.json").exists() else []
    done = [s for s in scores if s.get("status") == "done" and s.get("text_mu") is not None and s.get("ckpt") != "step-0"]
    best = max(done, key=lambda s: s["text_mu"]) if done else None
    kind = next((k for k in SCREEN_COMPS if run.name.startswith(k + "_")), "text")
    primary = kind.upper() if kind in SCREEN_COMPS else None
    screens = run_screens(run, primary) if primary else []
    if primary:  # MAIN / MATH / CODE run: best = best screen of its own competition vs that king (bar 0.003 / 0.002)
        args.target = BARS[primary]
        scored = [r for r in screens if r["ckpt"] != "start" and r.get(primary, {}).get("mu") is not None]
        top = max(scored, key=lambda r: r[primary]["mu"]) if scored else None
        best = {"ckpt": top["ckpt"], "text_mu": top[primary]["mu"], "lcb": top[primary].get("lcb")} if top else None
    if kind == "main":  # the MAIN trainer's val has one split
        val = [{**v, "val_split": {"all sources": x for x in (v.get("val_split") or {}).values()}} for v in val]
    status_map = pm2_status()
    st = status_map.get(run.name)
    tl = run_timeline(run, a)
    if (run / "final").exists() or (train and train[-1]["step"] >= steps):
        status = "finished"
    elif st == "online":
        status = "paused" if (run / "PAUSED").exists() else ("running" if train or val else "busy")
    elif tl["phase"] == "error in the trainer log":
        status = "crashed"
    elif train or val:
        status = "stopped"
    else:
        status = "idle"
    if status in ("stopped", "idle") and cmd(["pgrep", "-f", "logs/after_math.sh"]).strip():  # a queued MAIN run waits for the GPUs
        status = "queued"
        tl["phase"] = "queued: the MATH full check (A/B/C vs #43) holds the GPUs; MAIN run m3 starts automatically when it ends"
        tl["detail"] = f"{run.name} (shown) was stopped; the next run appears here as soon as it writes its first log line"
    campaign = merge_campaign()
    if campaign.get("phase") and status in ("idle", "stopped"):
        status = "merging"
        tl["phase"] = campaign["phase"]
        tl["detail"] = "GPUs are on the public-model merge search"
    gpus = gpu_stats()
    hw = f"{len(gpus)}× {gpus[0]['name']} {gpus[0]['mem_total'] / 1024:.0f} GiB" if gpus else "GPUs unknown"
    val_pool = a.get("val", "").rstrip("/").split("/")[-1] or "?"
    val_n = next((m.group(1) for _, msg in log_lines(run.name) if (m := re.search(r"\bval (\d+) seqs", msg))), a.get("val-n", "512"))
    pool = "+".join(x.rstrip("/").split("/")[-1] for x in a.get("pool", "").split(",") if x) or "?"
    data = {
        "run": run.name, "recipe": args.recipe or recipe, "schedule": schedule, "steps": steps,
        "save_every": int(a.get("save-every", 100)), "target": args.target, "clip": a.get("clip", "1.0"), "train_every": every,
        "train": train, "val": val, "scores": scores, "best_score": best, "status": status, "now": tl, "kind": kind, "screens": screens,
        "primary": primary, "screen_comps": SCREEN_COMPS.get(kind, []), "bars": BARS, "king_names": king_names() if primary else {},
        "eval_every": int(a.get("eval-every", 0) or 0), "screen_every": screen_interval(run, a),
        "tokens_per_step": tok_step, "hardware": f"{hw} · FSDP · {'LoRA adapters' if int(a.get('lora-r', 0)) > 0 else trained_part(a)}",
        "pool": pool, "valset": f"{val_pool} ({val_n} seqs)", "log_path": str(run / "train_log.jsonl"),
        "runs": all_runs(status_map), "evals": evaluations(), "campaign": campaign, "eval_jobs": eval_jobs(), "validator": validator_duel(), "gpus": gpus, "jobs": background_jobs(),
        "autopilot": autopilot(run, train), "rendered_epoch": time.time(),
    }
    blob = json.dumps(data)
    html = TEMPLATE.replace("__NAME__", args.name).replace("__TARGET__", f"{args.target:.4f}").replace("__DATA__", blob.replace("</", "<\\/"))
    for path, text in ((args.out, html), (args.out.with_name("data.json"), blob)):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text)
        os.replace(tmp, path)  # the polling page never reads a half-written file
    print(f"{args.out}: {len(train)} train rows, {len(val)} val rows, {len(scores)} scores, status {status}, {tl['phase']}")


if __name__ == "__main__":
    main()
