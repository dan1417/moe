#!/usr/bin/env bash
# All of today's kings in one round (user 21:20): 8 recipes on KTEXT, MAIN vs KMAIN / MATH vs KMATH / TEXT vs KTEXT,
# 4 800 seqs of sample A, both replica sets (no trainer running). Holds the GPU screen lock king_merge respects.
cd /root/workspace/MoE && source /data/teutonic/env.sh
L=/data/teutonic/gpu_screen.lock
while [[ -e $L ]]; do sleep 30; done
echo allkings > $L; trap 'rm -f $L' EXIT
PARITY_MERGE_SLOTS=8 python scripts/merge/campaign43.py --recipe /root/workspace/MoE/configs/merge/allkings/ak_kt_math050.json --recipe /root/workspace/MoE/configs/merge/allkings/ak_kt_ours_math025.json --recipe /root/workspace/MoE/configs/merge/allkings/ak_kt_ours_math050.json --recipe /root/workspace/MoE/configs/merge/allkings/ak_kt_mom100.json --recipe /root/workspace/MoE/configs/merge/allkings/ak_kt_mom200.json --recipe /root/workspace/MoE/configs/merge/allkings/ak_kt_mom100_ours_math025.json --recipe /root/workspace/MoE/configs/merge/allkings/ak_kt_ours2.json --recipe /root/workspace/MoE/configs/merge/allkings/ak_kt_dc1_math050.json --king main=KMAIN_c7ef5487:0.003 --king math=KMATH_ddd9e626:0.002 --king text=KTEXT_9a39afee:0.002 --comps main,math,text --campaign allkings --limit 4800 --sample tag:A --replicas 4 --batch-size 32 --pingpong 1 2>&1 | grep --line-buffered -E '^=== |  (MAIN|MATH|TEXT) |Traceback|Error|scored'
echo "ALLKINGS DONE"
