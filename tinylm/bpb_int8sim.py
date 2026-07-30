"""
Bits-per-byte of a checkpoint under SIMULATED Q8_0 int8 quantization.

Mirrors export.py version2_export / runq.c numerics in PyTorch:
- every Linear weight and the (tied) embedding are quantize-dequantized to
  symmetric int8 [-127,127] in groups of `group_size` (static, like the export)
- Linear INPUT activations are quantize-dequantized the same way per forward
  pass (dynamic, like runq.c's quantize() before each matmul)
- rmsnorm weights, RoPE, softmax and the KV cache stay fp32 (as in runq.c)

The int32 accumulation of runq.c is exact, so quant-dequant + fp32 matmul is a
faithful stand-in; residual difference is fp32 summation order only.

Usage:
    python bpb_int8sim.py --out_dir out_tok_512dep --vocab_size 512
    python bpb_int8sim.py --out_dir out --vocab_size 512 --no_act_quant   # weights-only
"""
import argparse
import json
import math
import os
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tinystories import Task, tok_name
from model import ModelArgs, Transformer

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", type=str, required=True)
ap.add_argument("--vocab_source", type=str, default="custom", choices=["custom", "llama2"])
ap.add_argument("--vocab_size", type=int, required=True)
ap.add_argument("--group_size", type=int, default=64)
ap.add_argument("--eval_iters", type=int, default=200)
ap.add_argument("--batch_size", type=int, default=128)
ap.add_argument("--device", type=str, default="cuda")
ap.add_argument("--no_act_quant", action="store_true", help="quantize weights only")
ap.add_argument("--fp32_baseline", action="store_true", help="no quantization: fp32 reference in this same harness")
args = ap.parse_args()

DATA = "data"


def qdq(t, group_size):
    """Quantize-dequantize a tensor Q8_0-style: int8 symmetric per group."""
    orig_shape = t.shape
    flat = t.reshape(-1, group_size)
    scale = flat.abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(flat / scale), -127, 127)
    return (q * scale).reshape(orig_shape)


class QuantLinear(nn.Module):
    """Linear with Q8_0-simulated weight and (optionally) input activations."""

    def __init__(self, lin, group_size, act_quant):
        super().__init__()
        self.weight = nn.Parameter(qdq(lin.weight.data, group_size), requires_grad=False)
        self.group_size = group_size
        self.act_quant = act_quant

    def forward(self, x):
        if self.act_quant:
            x = qdq(x, self.group_size)
        return F.linear(x, self.weight)


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

    gs = args.group_size
    act = not args.no_act_quant
    if args.fp32_baseline:
        print("fp32 baseline: no quantization applied")
        return evaluate(model, margs, "fp32 reference")
    # embedding: quantize-dequantize the table (runq dequantizes one row per step)
    model.tok_embeddings.weight.data = qdq(model.tok_embeddings.weight.data, gs)
    n_quant = 1
    for layer in model.layers:
        a, f = layer.attention, layer.feed_forward
        a.wq = QuantLinear(a.wq, gs, act); a.wk = QuantLinear(a.wk, gs, act)
        a.wv = QuantLinear(a.wv, gs, act); a.wo = QuantLinear(a.wo, gs, act)
        f.w1 = QuantLinear(f.w1, gs, act); f.w2 = QuantLinear(f.w2, gs, act)
        f.w3 = QuantLinear(f.w3, gs, act)
        n_quant += 7
    # classifier: tied to the embedding in these models; wrap so activations
    # into it are also quantized, using the already-qdq'd embedding table
    tied = model.output.weight.data_ptr() == model.tok_embeddings.weight.data_ptr()
    model.output = QuantLinear(model.output, gs, act)
    if tied:
        model.output.weight = nn.Parameter(model.tok_embeddings.weight.data, requires_grad=False)
    n_quant += 1
    model.to(device)
    print(f"quantized {n_quant} weight tensors, group_size={gs}, act_quant={act}, tied_classifier={tied}")
    return evaluate(model, margs, "Q8_0 simulated")


def evaluate(model, margs, label):
    device = next(model.parameters()).device
    max_seq_len = margs["max_seq_len"]
    iter_batches = partial(
        Task.iter_batches, batch_size=args.batch_size, max_seq_len=max_seq_len,
        vocab_size=args.vocab_size, vocab_source=args.vocab_source,
        device=device, num_workers=0,
    )
    losses = torch.zeros(args.eval_iters)
    bi = iter_batches(split="val")
    with torch.no_grad():
        for k in range(args.eval_iters):
            X, Y = next(bi)
            model(X, Y)  # fp32 forward: quantization error is the variable here
            losses[k] = model.last_loss.item()
    loss_nats = losses.mean().item()

    # tokens/byte on the val shard (same as bits_per_byte.py)
    val_bin = os.path.join(DATA, tok_name(args.vocab_size), "data00.bin")
    n_tokens = np.memmap(val_bin, dtype=np.uint16, mode="r").size
    with open(os.path.join(DATA, "TinyStories_all_data", "data00.json"), encoding="utf-8") as f:
        stories = json.load(f)
    n_bytes = sum(len(ex["story"].strip().encode("utf-8")) for ex in stories)
    tok_per_byte = n_tokens / n_bytes
    bpb = loss_nats * tok_per_byte / math.log(2)

    print(f"out_dir           : {args.out_dir} ({label})")
    print(f"val loss (nats/tok): {loss_nats:.4f}")
    print(f"tokens/byte (val) : {tok_per_byte:.4f}")
    print(f">>> BITS/BYTE     : {bpb:.4f}")


if __name__ == "__main__":
    main()
