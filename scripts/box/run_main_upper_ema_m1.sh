#!/usr/bin/env bash
cd /root/workspace/MoE
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 TEUTONIC_EVAL_CACHE=/data/teutonic/eval_cache
# compile caches on local disk: 8 ranks compiling flex-attention kernels into ~/.triton on a FUSE /root race
# (rank sees the .so path before it is readable: 'cuda_utils...so: cannot open shared object file', B200 box)
export TRITON_CACHE_DIR=/data/teutonic/cache/triton TORCHINDUCTOR_CACHE_DIR=/data/teutonic/cache/inductor
# full-parameter runs sit at the allocator ceiling (B200 batch 4: 161 GiB allocated, 176.6 GiB reserved -> cudaMalloc
# retries, 132 s/step instead of 75 s); expandable segments keep reserved close to allocated
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec /root/workspace/MoE/.venv-eval/bin/python -m torch.distributed.run --nproc_per_node 8 --master_port 29500 scripts/train/cpt_fsdp.py \
  --init /data/teutonic/ckpt/start --pool /data/teutonic/pools/text_cpt_v1 --val /data/teutonic/pools/text_val_v1 \
  --out /data/teutonic/ckpt/main_upper_ema_m1 --steps 800 --lr 1e-4 --warmup 50 --min-lr-frac 0.1 \
  --lora-r 16 --lora-alpha 32 --optim adamw --eval-every 50 --val-n 512 --save-every 100 --init /data/teutonic/ckpt/start_m1 --pool /data/teutonic/pools/text_cpt_v10 --attn flex_attention --lr 5e-5 --warmup 10 --min-lr-frac 0.5 --steps 200 --eval-every 20 --save-every 50 --freeze \\.mlp\\.gate\\.\|\^model\\.embed_tokens\\.\|\^model\\.layers\\.\(\[0-9\]\|1\[0-5\]\)\\. --lora-r 0 --dtype fp32 --optim adamw8bit --offload-opt 1 --lr 1.5e-6 --lr-dense-frac 2.0 --warmup 20 --min-lr-frac 1.0 --ema 0.98 --val-raw 0 --contract-dir /data/teutonic/models/95ef7a26303df937d0dbcd2bbc2fe8d54b6acab124f552416d2988ae697091d0 --save-every 25 --eval-every 25 --steps 600 --pool /data/teutonic/pools/main_cpt_v1\,/data/teutonic/pools/main_cpt_v2\,/data/teutonic/pools/main_cpt_v3\,/data/teutonic/pools/main_cpt_v4\,/data/teutonic/pools/main_cpt_v5\,/data/teutonic/pools/main_cpt_v6 --val /data/teutonic/pools/main_val_v1 --batch 16 --accum 16
