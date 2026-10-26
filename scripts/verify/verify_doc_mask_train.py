#!/usr/bin/env python3
"""Check the trainer's document masking (cpt_fsdp.py --doc-mask 1) against the validator's definition.

  python scripts/verify/verify_doc_mask_train.py --init <ckpt> [--attn flex_attention] [--layers 2]

On a truncated model (first --layers decoder layers, one GPU):
1. layout parity: positions / valid targets equal teutonic.evaluator.document_masking.document_layout, and the
   per-row target counts equal teutonic.evaluation.masking.scored_token_counts;
2. isolation: a row packing several EOS-terminated fragments gives, per fragment, the same CE sum as that fragment
   scored alone (no context from the earlier fragments, positions restarting at 0);
3. backward runs through the masked path (flex attention compiles with the packed-sequence mask).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/train"))
sys.path.insert(0, str(REPO))

import cpt_fsdp  # noqa: E402

EOS = cpt_fsdp.DOCUMENT_EOS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True, type=Path)
    ap.add_argument("--attn", default="flex_attention")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from teutonic.evaluation.masking import scored_token_counts
    from teutonic.evaluator.document_masking import document_layout

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    model = cpt_fsdp.build_model(args.init, torch.bfloat16, 0, args.attn, debug_layers=args.layers).to(device)
    model.eval()
    vocab = model.config.vocab_size

    # fragments: lengths sum to 2048, each ends with EOS except the last (a window ends mid-document); one doubled EOS
    lengths = [700, 1, 300, 1047]
    frags = []
    for i, n in enumerate(lengths):
        body = torch.randint(1000, 100000, (n,))
        if i < len(lengths) - 1:
            body[-1] = EOS
        frags.append(body)
    row = torch.cat(frags)[None].to(device)
    assert row.shape[1] == 2048

    # 1. layout parity with the validator
    positions, valid = cpt_fsdp.document_layout(row)
    ref = document_layout(row)
    assert torch.equal(positions, ref.positions), "positions differ from the validator's layout"
    assert torch.equal(valid, ref.valid_targets), "valid targets differ from the validator's layout"
    assert int(valid.sum()) == scored_token_counts(row.tolist())[0], "target count differs from scored_token_counts"
    print(f"layout parity ok: {int(valid.sum())} scored targets of {row.shape[1] - 1}")

    # 2. isolation: packed fragments == fragments scored alone
    with torch.no_grad():
        total, count = cpt_fsdp.seq_ce(model, row, doc_mask=True)
        alone_sum, alone_count = 0.0, 0
        for frag in frags:
            if len(frag) < 2:
                continue  # a 1-token fragment has no target
            s, c = cpt_fsdp.seq_ce(model, frag[None].to(device), doc_mask=True)
            alone_sum += float(s)
            alone_count += int(c)
        unmasked, n_all = cpt_fsdp.seq_ce(model, row, doc_mask=False)
    packed = float(total) / float(count)
    alone = alone_sum / alone_count
    print(f"packed masked CE {packed:.6f} | fragments alone {alone:.6f} | diff {packed - alone:+.2e} "
          f"(targets {int(count)} vs {alone_count}) | unmasked row {float(unmasked) / float(n_all):.6f}")
    assert int(count) == alone_count, "packed and separate target counts differ"
    assert abs(packed - alone) < 2e-3, "packed fragments see context across EOS (masking not applied)"

    # 2b. perturbation: rewrite the first fragment; with isolation the later fragments' token losses cannot move
    def token_losses(ids):
        positions, _ = cpt_fsdp.document_layout(ids)
        with torch.no_grad():
            logits = model(input_ids=ids, position_ids=positions, use_cache=False).logits
        return torch.nn.functional.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none")

    other = row.clone()
    other[0, : lengths[0] - 1] = torch.randint(1000, 100000, (lengths[0] - 1,), device=device)
    later = slice(lengths[0] + lengths[1], row.shape[1] - 1)  # targets inside fragments 3 and 4
    moved = (token_losses(row)[later] - token_losses(other)[later]).abs().max().item()
    print(f"perturbing fragment 1 moves later fragments' token losses by at most {moved:.2e}")
    assert moved < 1e-2, "later fragments depend on the first fragment: attention crosses EOS"

    # 3. backward through the masked path
    # the trainer stays in eval() too (no dropout; MiMo noaux_tc routing refuses train mode) - gradients flow regardless
    s, c = cpt_fsdp.seq_ce(model, row, doc_mask=True)
    (s.sum() / c.sum()).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    norm = torch.sqrt(sum(g.float().pow(2).sum() for g in grads))
    print(f"backward ok: {len(grads)} grads, |g| {float(norm):.4f}")
    print("PASS")


if __name__ == "__main__":
    main()
