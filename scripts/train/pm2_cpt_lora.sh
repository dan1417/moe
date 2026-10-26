#!/usr/bin/env bash
# Run the LoRA continued-pretraining run (cpt_fsdp.py, 8 GPUs) under pm2 so it survives the shell and can be followed with
#   pm2 logs <name> --lines 50      pm2 stop <name>      python scripts/train/dashboard.py --run /data/teutonic/ckpt/<name> ...
#   bash scripts/train/pm2_cpt_lora.sh text_lora_a --batch 4 --accum 8 [more cpt_fsdp.py args]
# No auto-restart: a crashed 110B run must be looked at, not relaunched (resume: --init <run>/step-N --warmup 0).
set -euo pipefail
NAME=${1:?pm2 process name = run name}; shift
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=$PWD/.venv-eval/bin/python
ARGS=$(printf ' %q' "$@")   # quoted: regex args (--train-only) contain ( | [ characters
mkdir -p /data/teutonic/logs /data/teutonic/ckpt
cat > "/data/teutonic/logs/run_${NAME}.sh" <<RUN
#!/usr/bin/env bash
cd $PWD
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 TEUTONIC_EVAL_CACHE=/data/teutonic/eval_cache
# compile caches on local disk: 8 ranks compiling flex-attention kernels into ~/.triton on a FUSE /root race
# (rank sees the .so path before it is readable: 'cuda_utils...so: cannot open shared object file', B200 box)
export TRITON_CACHE_DIR=/data/teutonic/cache/triton TORCHINDUCTOR_CACHE_DIR=/data/teutonic/cache/inductor
# full-parameter runs sit at the allocator ceiling (B200 batch 4: 161 GiB allocated, 176.6 GiB reserved -> cudaMalloc
# retries, 132 s/step instead of 75 s); expandable segments keep reserved close to allocated
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec $PY -m torch.distributed.run --nproc_per_node 8 --master_port 29500 scripts/train/cpt_fsdp.py \\
  --init /data/teutonic/ckpt/start --pool /data/teutonic/pools/text_cpt_v1 --val /data/teutonic/pools/text_val_v1 \\
  --out /data/teutonic/ckpt/${NAME} --steps 800 --lr 1e-4 --warmup 50 --min-lr-frac 0.1 \\
  --lora-r 16 --lora-alpha 32 --optim adamw --eval-every 50 --val-n 512 --save-every 100$ARGS
RUN
chmod +x "/data/teutonic/logs/run_${NAME}.sh"
pm2 delete "$NAME" >/dev/null 2>&1 || true
pm2 start bash --name "$NAME" --no-autorestart --time --merge-logs \
  --output "/data/teutonic/logs/pm2_${NAME}.out.log" --error "/data/teutonic/logs/pm2_${NAME}.err.log" \
  -- "/data/teutonic/logs/run_${NAME}.sh"
pm2 save >/dev/null 2>&1 || true
pm2 ls
