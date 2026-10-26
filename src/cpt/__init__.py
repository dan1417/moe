"""Full-parameter continued pretraining of the Teutonic king (8×B300 target).

Design (see docs/CPT.md):
  * locked MiMo-V2 remote code; MoE experts re-hosted as stacked grouped-GEMM weights
  * FSDP2 per decoder layer, fp32 master weights, bf16 compute, fp32 reductions
  * AdamW with routers frozen and pure next-token CE (no aux / z-loss / bias updates)
  * EMA of weights on host memory; exports are validated with src/parity before submission
"""
