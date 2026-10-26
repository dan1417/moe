#!/usr/bin/env bash
# User 19:15 JST Oct 11: "merge our checkpoints with the current top models, every possible way, best model per competition".
# Wait until king_merge has ingested the new CODE king (KCODE_0c681d78, 5Ek5Ko v56), stop king_merge (its old-style MAIN/MATH
# round would take the GPUs between our rounds), release the GPU lock, run merge_search3 on all 8 GPUs.
cd /root/workspace/MoE
until python3 -c "import json,sys; sys.exit(0 if 'KCODE_0c681d78' in json.load(open('configs/merge/k41/models.json')) else 1)"; do sleep 15; done
echo "$(date -u +%H:%M:%S) KCODE_0c681d78 registered: $(python3 -c "import json; print(json.load(open('configs/merge/k41/models.json'))['KCODE_0c681d78'])")"
sleep 20; pm2 stop king_merge >/dev/null 2>&1
rm -f /data/teutonic/gpu_screen.lock
exec .venv-eval/bin/python scripts/merge/merge_search3.py --name all3 --max-rounds 8
