#!/usr/bin/env bash
# Watch the shared account (token in /data/teutonic/hf_token_share = yamalies) every 15 min for checkpoints that are not our own
# backups (repos recorded in our uploads.json files, plus everything that existed at 01:50 JST Oct 12); download each new one
# into /data/teutonic/models/shared/<repo> and register it as SH_<name> in models.json for merging (user 01:45 JST Oct 12).
cd /root/workspace/MoE
exec .venv-eval/bin/python - <<'PY'
import glob, json, time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
tok = open("/data/teutonic/hf_token_share").read().strip()
api = HfApi(token=tok)
owners = ["Valhalla-ai", api.whoami()["name"]]  # the team org (user 02:00 JST Oct 12) and the account
def ours():
    s = {"yamalies/ced06b7a2131"}
    for f in glob.glob("/data/teutonic/ckpt/*/uploads.json") + ["/data/teutonic/ckpt/uploads.json", "/data/teutonic/submit/uploads.json"]:
        try:
            u = json.load(open(f)); items = u if isinstance(u, list) else list(u.values())
            s |= {x.get("repo") for x in items if isinstance(x, dict)}
        except Exception:
            pass
    return s
known = {m.id for o in owners for m in api.list_models(author=o, limit=1000)}
print(time.strftime("%H:%M:%S", time.gmtime()), f"watching {', '.join(owners)}: {len(known)} repos known", flush=True)
while True:
    time.sleep(900)
    try:
        cur = [m for o in owners for m in api.list_models(author=o, limit=1000, full=True)]
    except Exception as e:
        print("list failed:", str(e)[:120], flush=True); continue
    mine = ours()
    for m in cur:
        if m.id in known:
            continue
        known.add(m.id)
        if m.id in mine:
            continue  # our own new backup
        print(time.strftime("%H:%M:%S", time.gmtime()), f"NEW SHARED CHECKPOINT {m.id}", flush=True)
        try:
            files = api.list_repo_files(m.id)
            if "model.safetensors.index.json" not in files:
                print(f"  {m.id}: no safetensors index, not a model checkpoint", flush=True); continue
            local = Path("/data/teutonic/models/shared") / m.id.replace("/", "__")
            snapshot_download(m.id, local_dir=str(local), token=tok, max_workers=32)
            mj = Path("configs/merge/k41/models.json"); M = json.loads(mj.read_text())
            M["SH_" + m.id.split("/")[-1]] = str(local); mj.write_text(json.dumps(M, indent=1) + "\n")
            print(time.strftime("%H:%M:%S", time.gmtime()), f"DOWNLOADED {m.id} -> {local} (SH_{m.id.split('/')[-1]})", flush=True)
        except Exception as e:
            print(f"  download of {m.id} failed: {str(e)[:200]}", flush=True)
PY
