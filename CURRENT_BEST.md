# CURRENT_BEST.md — where the current best model lives

## Multi-competition campaign (2026-10-07) — CODE submission: `t1_d36050_pe015` as `teutonic-II-110B-A7B-001`

**CODE KING #1 since 2026-10-07 20:20 UTC** (duel accepted 20:17: mu +0.002486, LCB +0.002387 on 30k vs #41). Submission (19:05 UTC): `t1_d36050_pe015` = T1 + 0.5·D36 + experts-only momentum 0.15·(T1 − #40); confirmed +0.00289 /
LCB +0.00251 on the exact code sample; model digest `7e4aa823…`, hotkey `sn3auto002` (uid 110). Recipe
`configs/merge/k41/t1_d36050_pe015.json`; written dir `/data/teutonic/submit/t1_d36050_pe015`.
The first attempt (`t1_d36050` as `…-t1d36050`, hotkey `sn3auto001`) FAILED: a second upload into the same prefix tripped the
250 GB quota and the controller wiped the prefix before `ready` (see 41.md 18:45). Never upload twice into a prefix.

### Earlier notes (first attempt)

**Model:** `t1_d36050` = TEXT king T1 (`teutonic-II-110B-A7B-5DALwjE4-93`, `95ef7a26…`) + 0.5·D36, D36 = #36 − #35
(5Ek5Ko's step) as an int8 row-quantized task vector in T1/#41's hidden basis (`/data/teutonic/vec/d36_35q`).
- vs MAIN king #41 on the **CODE** mix (70/15/15): screen +0.00277 (2400 mix seqs), **confirmed +0.00284 / LCB +0.00251
  on 3000 seqs of an independent exact code-competition sample** (bar 0.002). MATH mix +0.00202 (bar 0.002, marginal),
  MAIN +0.00275 (bar 0.003), TEXT vs T1 +0.00061.
- Written: `/data/teutonic/submit/t1_d36050` (220.6 GB, 42 shards, `verify_submission.py` READY, safetensors digest
  `396f33b5a577a7e88cf18b11cb8a2b044e28b92d63de373792e64fd22060c107`); re-scored = recipe. **Submitted to the CODE competition as `teutonic-II-110B-A7B-t1d36050`** (hotkey `5FiKPA7d…` uid 4, model digest
  `29611fc4…`, `ready` finalized 2026-10-07 18:15 UTC).
- Recipe (exact): `configs/merge/k41/t1_d36050.json`; full log and every other recipe: [41.md](41.md).
- Runner-up: `t1_d36050_pe015` (+ experts-only momentum 0.15·(T1 − #40)): CODE +0.00291, MATH +0.00212, MAIN +0.00285 on
  the screen; written to `/data/teutonic/submit/t1_d36050_pe015` (unconfirmed).

Rebuild: `source /data/teutonic/env.sh && python scripts/merge/write_merge.py --recipe configs/merge/k41/t1_d36050.json --out <dir> --gpus cpu`
(needs #41, T1 and `d36_35q`: `python scripts/merge/remote_delta40.py --a 36 --b 35 --ref $T1 --out /data/teutonic/vec/d36_35q --int8`).

---

## King #40 campaign (2026-10-05) — current best: `f_d36050_pe015`

**Model:** `f_d36050_pe015` = king #40 (`teutonic-II-110B-A7B-5DALwjE4-88`) + 0.5·D36 (all tensors)
+ 0.15·(#40 − #36) on the routed experts only.
- D36 = #36 − #35 (5Ek5Ko's step, the last non-5DALwjE4 step, partly washed out by later training);
  (#40 − #36) = 5DALwjE4's 4-step path (k38-style "momentum", experts only). Both stored as int8 row-quantized task
  vectors in #40's hidden basis (`/opt/teutonic/vec/d36_35q`, `/opt/teutonic/vec/p40_36q`).
- **mu_hat +0.00112 vs king #40** (LCB +0.00080, 800 seqs of tag:A; code +0.0012, math +0.0006, pretrain +0.0015,
  sft +0.0015, txt360 +0.0012); **confirmed on 3000 seqs: +0.00102 (LCB +0.00085)**, all sources positive.
  Crown needs ≈ +0.0036. Full log: [40.md](40.md).
- **HF (public, gated = manual approval, no README): `yamalies/e2q5ngla76um`** — 42 shards, 220.6 GB, every shard's
  LFS sha256 checked after upload; contract files checked against chain.toml.
- Previous upload: `up_d36_050q` (#40 + 0.5·D36, +0.00088) → `yamalies/myzxnk3nvdm6`.
- Recipe (exact, scored = shipped): `configs/merge/k40/f_d36050_pe015.json`; uploads: `configs/merge/k40/uploads.json`.

Rebuild (needs king #40 and the two int8 deltas; deltas are rebuilt from R2 in ~30 min each):
```bash
source .venv-eval/bin/activate
python scripts/merge/remote_delta40.py --a 36 --b 35 --out /opt/teutonic/vec/d36_35q --int8      # D36
python scripts/merge/remote_delta40.py --a 40 --b 36 --a-local <king40 dir> --out /opt/teutonic/vec/p40_36q --int8
python scripts/merge/stream_upload40.py --recipe configs/merge/k40/f_d36050_pe015.json            # or write_merge.py
```

---

## Previous: king #31 campaign

**Model:** `L1_coef_g060` (king #31 merge campaign, 2026-09-26)
- #31u + Dm with learned per-tensor coefficients for Dm and S31 (one gradient step), where
  Dm = k30u − #30u (public-king merge 0.17·D26 + 0.13·D25) and S31 = #31u − #30u. Details: [31.md](31.md), README.
- **mu_hat +0.00185 vs king #31** (LCB +0.0014, 800 seqs of tag:A); the plain king merge scores +0.00166 and the crown
  needs ≈ +0.0036, so this model will **not** take the crown by itself. It's the best starting point for training.
- 220.6 GB bf16, 42 safetensors shards, in the #29/k30u hidden basis. `verify_submission.py`: READY.
  Safetensors digest `15d7a290d3c2333f99c3b5ad5b65fb38ed31aa9cf6f163c3d6519b50f1ec1ec4`.
- Exact recipe: `configs/merge/k31/L1_coef_g060.json` (`ship` block → `scripts/merge/write_merge.py`).

## Storage: split over four private HF repos (each account's private quota is 100 GB)

| part | repo (private) | shards | size |
|---|---|---|---:|
| 0 | `jenny08311/x4s9irot5g92` | model-00001 … 00011 (+ config/tokenizer/code) | 59.0 GB |
| 1 | `scoutminer/ifm52afelx82` | model-00012 … 00021 (+ config/tokenizer/code) | 53.6 GB |
| 2 | `touristAI/im5vv2qo7xuj` | model-00022 … 00031 (+ config/tokenizer/code) | 53.6 GB |
| 3 | `yamalies/ywhizwyvuks0` | model-00032 … 00042 (+ config/tokenizer/code) | 54.3 GB |

The machine-readable layout plus the size and sha256 of every file is in `configs/best/current_best.json`.
**Tokens are not stored in git.** Keep them yourself and pass them as `HF_TOKENS` or a tokens file.

> Upload status: **uploaded 2026-09-26 14:31 UTC**. All four repos are private, and every shard's size and
> LFS sha256 on HF match `configs/best/current_best.json`. The local copy is at `/opt/teutonic/submit/im5vv2qo7xuj`
> (split into `/opt/teutonic/submit/parts/p0..p3` as hard links). To re-upload (resumable):
> ```bash
> source /opt/teutonic/env.sh
> nohup python scripts/data/upload_best.py --tokens-file /root/.huggingface/best_split_tokens \
>     > /opt/teutonic/logs/upload_best.log 2>&1 &
> ```

## Download anywhere (with this repo checked out)

```bash
pip install -U huggingface_hub
HF_TOKENS=hf_tok1,hf_tok2,hf_tok3,hf_tok4 \
    python scripts/data/download_best.py --out /root/best_model     # downloads all 4 parts, then verifies sha256
```

The script matches each repo to the token of its owner, downloads the four parts in parallel into one directory,
and checks every file against the manifest.

## Rebuild without HF (needs king #31, king #30 and k30u in the common basis)

```bash
python - <<'EOF'
import json; r = json.load(open("configs/merge/k31/L1_coef_g060.json"))
json.dump(r["ship"], open("/tmp/L1_coef_g060.ship.json", "w"))
EOF
python scripts/merge/write_merge.py --recipe /tmp/L1_coef_g060.ship.json --out /opt/teutonic/submit/L1_coef_g060
```
