"""
Compute BITS-PER-BYTE for a trained checkpoint.

Validation cross-entropy (nats/token) is NOT comparable across tokenizers, because
each tokenizer chops the text into a different number of tokens. Bits-per-byte
normalizes that away, so a char-level model and a BPE model can be ranked fairly:

    bpb = mean_val_loss_nats/token  x  (val_tokens / val_bytes)  /  ln(2)

Usage:
    python bits_per_byte.py --out_dir out --vocab_source custom --vocab_size 512
    python bits_per_byte.py --out_dir out_char --vocab_source custom --vocab_size 105
"""
import argparse
import glob
import json
import math
import os
from functools import partial

import numpy as np
import torch

from tinystories import Task, tok_name

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", type=str, required=True, help="dir containing ckpt.pt")
ap.add_argument("--vocab_source", type=str, default="custom", choices=["custom", "llama2"])
ap.add_argument("--vocab_size", type=int, required=True)
ap.add_argument("--eval_iters", type=int, default=200)
ap.add_argument("--batch_size", type=int, default=128)
ap.add_argument("--device", type=str, default="cuda")
ap.add_argument("--looped", action="store_true", help="checkpoint is a looped transformer")
args = ap.parse_args()

if args.looped:
    from model_looped import LoopedModelArgs as ModelArgs, LoopedTransformer as Transformer
else:
    from model import ModelArgs, Transformer

DATA = "data"


def val_tokens_per_byte(vocab_source, vocab_size):
    """Exact tokens/byte on the validation shard (shard 0)."""
    if vocab_source == "llama2":
        val_bin = os.path.join(DATA, "TinyStories_all_data", "data00.bin")
    else:
        val_bin = os.path.join(DATA, tok_name(vocab_size), "data00.bin")
    n_tokens = np.memmap(val_bin, dtype=np.uint16, mode="r").size
    with open(os.path.join(DATA, "TinyStories_all_data", "data00.json"), encoding="utf-8") as f:
        stories = json.load(f)
    n_bytes = sum(len(ex["story"].strip().encode("utf-8")) for ex in stories)
    return n_tokens, n_bytes


def main():
    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=device)
    margs = ckpt["model_args"]
    model = Transformer(ModelArgs(**margs))
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.to(device).eval()

    max_seq_len = margs["max_seq_len"]
    iter_batches = partial(
        Task.iter_batches, batch_size=args.batch_size, max_seq_len=max_seq_len,
        vocab_size=args.vocab_size, vocab_source=args.vocab_source,
        device=device, num_workers=0,
    )

    # mean per-token cross-entropy on val (nats), averaged over eval_iters batches
    losses = torch.zeros(args.eval_iters)
    bi = iter_batches(split="val")
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" else \
        torch.autocast(device_type="cpu", enabled=False)
    with torch.no_grad():
        for k in range(args.eval_iters):
            X, Y = next(bi)
            with ctx:
                model(X, Y)
                losses[k] = model.last_loss.item()
    loss_nats = losses.mean().item()

    n_tokens, n_bytes = val_tokens_per_byte(args.vocab_source, args.vocab_size)
    tok_per_byte = n_tokens / n_bytes
    bpb = loss_nats * tok_per_byte / math.log(2)
    n_params = sum(p.numel() for p in model.parameters())

    print(f"out_dir           : {args.out_dir}")
    print(f"params            : {n_params:,}")
    print(f"vocab             : {args.vocab_source} / {args.vocab_size}")
    print(f"val loss (nats/tok): {loss_nats:.4f}")
    print(f"tokens/byte (val) : {tok_per_byte:.4f}  ({n_tokens:,} tok / {n_bytes:,} bytes)")
    print(f"bits/token        : {loss_nats/math.log(2):.4f}")
    print(f">>> BITS/BYTE     : {bpb:.4f}   <-- comparable across tokenizers")


if __name__ == "__main__":
    main()
