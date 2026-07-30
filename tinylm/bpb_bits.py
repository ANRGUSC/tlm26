"""
Bits-per-byte across a WEIGHT bit-width ladder (fp32 -> int8 -> int4 -> int2 ->
ternary), extending bpb_int8sim.py to a bits-per-weight axis.

Post-training quantization only -- no retraining. Each Linear weight and the
(tied) token-embedding table are quantize-dequantized per group of `group_size`;
a fp32 forward then measures the coherence cost as bits-per-byte, isolating
quantization error as the only variable (same faithful stand-in for runq.c that
bpb_int8sim.py validated for int8: int32 accumulation is exact, so qdq + fp32
matmul differs only in fp32 summation order).

Ladder levels (symmetric, per-group):
  fp32     : no quantization (reference, same harness)
  int8     : absmax scale, qmax=127     (matches runq.c Q8_0; ~free per section 9)
  int4     : absmax scale, qmax=7
  int2     : absmax scale, qmax=1  -> {-1,0,+1} by absmax
  ternary  : BitNet b1.58 absmean scale, q=clamp(round(w/mean|w|), -1, 1)

Weights-only by default (cleanest single variable: activations stay fp32 like
the KV cache / norms). Pass --act_bits 8 to add runq.c-style int8 activation
quant on top of the weight quant.

Run on CPU when a training job holds the GPU, so this does not disturb it:
  TS_TOK_NAME=tok512dep python bpb_bits.py --out_dir out_tok_512dep \
      --vocab_size 512 --device cpu --eval_iters 200 \
      --levels fp32,int8,int4,int2,ternary
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

from tinystories import Task, tok_name, get_tokenizer_model_path
from tokenizer import Tokenizer
from model import ModelArgs, Transformer

# nominal storage bits per weight for size reporting (ternary stored in 2 bits on
# a byte-addressable MCU, but reported at its information-theoretic 1.58)
NOMINAL_BITS = {"fp32": 32.0, "int8": 8.0, "int4": 4.0, "int2": 2.0, "ternary": 1.58}

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", type=str, required=True)
ap.add_argument("--vocab_source", type=str, default="custom", choices=["custom", "llama2"])
ap.add_argument("--vocab_size", type=int, required=True)
ap.add_argument("--group_size", type=int, default=64)
ap.add_argument("--eval_iters", type=int, default=200)
ap.add_argument("--batch_size", type=int, default=128)
ap.add_argument("--device", type=str, default="cpu")
ap.add_argument("--levels", type=str, default="fp32,int8,int4,int2,ternary")
ap.add_argument("--act_bits", type=int, default=0, help="0 = fp32 activations; 8 = runq-style int8 act quant")
ap.add_argument("--out_md", type=str, default="../docs/quant-ladder-results.md")
ap.add_argument("--sample", action="store_true", help="also generate a completion per level (coherence check)")
ap.add_argument("--sample_prompt", type=str, default="Once upon a time")
ap.add_argument("--sample_tokens", type=int, default=80)
ap.add_argument("--sample_temp", type=float, default=0.8)
args = ap.parse_args()

DATA = "data"


def qdq(t, group_size, level):
    """Quantize-dequantize a weight tensor per group at the given ladder level."""
    if level == "fp32":
        return t
    shape = t.shape
    flat = t.reshape(-1, group_size).float()
    if level == "ternary":
        scale = flat.abs().mean(dim=1, keepdim=True)                 # BitNet absmean
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = torch.clamp(torch.round(flat / scale), -1, 1)
        return (q * scale).reshape(shape).to(t.dtype)
    qmax = {"int8": 127, "int4": 7, "int2": 1}[level]
    scale = flat.abs().amax(dim=1, keepdim=True) / qmax              # absmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(flat / scale), -qmax, qmax)
    return (q * scale).reshape(shape).to(t.dtype)


class QuantLinear(nn.Module):
    """Linear with weights qdq'd at `level`, optional runq-style int8 act quant."""

    def __init__(self, lin, group_size, level, act_quant):
        super().__init__()
        self.weight = nn.Parameter(qdq(lin.weight.data, group_size, level), requires_grad=False)
        self.group_size = group_size
        self.act_quant = act_quant

    def forward(self, x):
        if self.act_quant:
            x = qdq(x, self.group_size, "int8")
        return F.linear(x, self.weight)


def build_quantized(out_dir, level, group_size, act_quant, device):
    """Load a fresh checkpoint and apply `level` weight quant. Returns (model, margs, n_wt)."""
    ckpt = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location=device)
    margs = ckpt["model_args"]
    model = Transformer(ModelArgs(**margs))
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.to(device).eval()

    n_wt = 0  # count of quantized weight elements (for size reporting)
    if level == "fp32":
        n_wt = model.tok_embeddings.weight.numel()
        for layer in model.layers:
            for m in (layer.attention.wq, layer.attention.wk, layer.attention.wv,
                      layer.attention.wo, layer.feed_forward.w1, layer.feed_forward.w2,
                      layer.feed_forward.w3):
                n_wt += m.weight.numel()
        return model, margs, n_wt

    model.tok_embeddings.weight.data = qdq(model.tok_embeddings.weight.data, group_size, level)
    n_wt += model.tok_embeddings.weight.numel()
    for layer in model.layers:
        a, f = layer.attention, layer.feed_forward
        for name in ("wq", "wk", "wv", "wo"):
            lin = getattr(a, name)
            n_wt += lin.weight.numel()
            setattr(a, name, QuantLinear(lin, group_size, level, act_quant))
        for name in ("w1", "w2", "w3"):
            lin = getattr(f, name)
            n_wt += lin.weight.numel()
            setattr(f, name, QuantLinear(lin, group_size, level, act_quant))
    # classifier tied to embedding: reuse the already-qdq'd table
    tied = model.output.weight.data_ptr() == model.tok_embeddings.weight.data_ptr()
    model.output = QuantLinear(model.output, group_size, level, act_quant)
    if tied:
        model.output.weight = nn.Parameter(model.tok_embeddings.weight.data, requires_grad=False)
    model.to(device)
    return model, margs, n_wt


def evaluate(model, margs, device):
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
            model(X, Y)
            losses[k] = model.last_loss.item()
    loss_nats = losses.mean().item()

    val_bin = os.path.join(DATA, tok_name(args.vocab_size), "data00.bin")
    n_tokens = np.memmap(val_bin, dtype=np.uint16, mode="r").size
    with open(os.path.join(DATA, "TinyStories_all_data", "data00.json"), encoding="utf-8") as f:
        stories = json.load(f)
    n_bytes = sum(len(ex["story"].strip().encode("utf-8")) for ex in stories)
    tok_per_byte = n_tokens / n_bytes
    bpb = loss_nats * tok_per_byte / math.log(2)
    return loss_nats, tok_per_byte, bpb


def sample_completion(model, enc, device):
    """Greedy-ish completion from a fixed prompt for a coherence eyeball."""
    ids = enc.encode(args.sample_prompt, bos=True, eos=False)
    x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
    with torch.no_grad():
        y = model.generate(x, args.sample_tokens, temperature=args.sample_temp, top_k=None)
    return enc.decode(y[0].tolist())


def main():
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    act_quant = args.act_bits == 8
    levels = [l.strip() for l in args.levels.split(",") if l.strip()]
    enc = None
    if args.sample:
        enc = Tokenizer(tokenizer_model=get_tokenizer_model_path(vocab_size=args.vocab_size))
    print(f"model={args.out_dir}  device={device}  group_size={args.group_size}  "
          f"act_quant={'int8' if act_quant else 'fp32'}  eval_iters={args.eval_iters}")
    print(f"tokenizer dir = data/{tok_name(args.vocab_size)}  (set via TS_TOK_NAME)\n")

    rows = []
    fp32_bpb = None
    for level in levels:
        model, margs, n_wt = build_quantized(args.out_dir, level, args.group_size, act_quant, device)
        loss, tpb, bpb = evaluate(model, margs, device)
        if level == "fp32":
            fp32_bpb = bpb
        bits = NOMINAL_BITS[level]
        # weight blob size + group-scale overhead (one fp16 scale per group)
        blob_kb = n_wt * bits / 8 / 1024
        if level != "fp32":
            blob_kb += n_wt / args.group_size * 2 / 1024  # fp16 scale per group
        d_bpb = None if fp32_bpb is None else bpb - fp32_bpb
        sample = sample_completion(model, enc, device) if args.sample else None
        rows.append((level, bits, n_wt, blob_kb, loss, tpb, bpb, d_bpb, sample))
        dstr = "" if d_bpb is None else f"  Dbpb={d_bpb:+.4f}"
        print(f"  {level:<8} bits/wt={bits:>5}  size={blob_kb:7.1f}KB  "
              f"val_loss={loss:.4f}  bpb={bpb:.4f}{dstr}")
        if sample:
            print(f"    sample: {sample!r}")

    write_md(rows, device, act_quant)


def write_md(rows, device, act_quant):
    lines = []
    lines.append("# Quantization ladder — bits-per-weight vs coherence\n")
    lines.append(f"*Model: `{args.out_dir}` ({rows[0][2]:,} quantized weight elements), "
                 f"tokenizer `{tok_name(args.vocab_size)}`, group_size {args.group_size}, "
                 f"activations {'int8' if act_quant else 'fp32'}, {args.eval_iters} val batches, "
                 f"device {device}. Post-training quantization only — no retraining.*\n")
    lines.append("Extends `bpb_int8sim.py` (RESEARCH_LOG §9) to a full bits-per-weight axis, "
                 "via `bpb_bits.py`. Every weight is quantize-dequantized per group and scored "
                 "on the same fp32 harness as every other model, so the delta is pure "
                 "quantization error.\n")
    lines.append("| Level | Bits/weight | Weight blob | Val loss (nats) | Bits/byte | Δ bpb vs fp32 |")
    lines.append("|---|---|---|---|---|---|")
    for level, bits, n_wt, blob_kb, loss, tpb, bpb, d_bpb, sample in rows:
        dstr = "—" if d_bpb is None else f"{d_bpb:+.4f}"
        lines.append(f"| {level} | {bits:g} | {blob_kb:.1f} KB | {loss:.4f} | {bpb:.4f} | {dstr} |")
    lines.append("")
    if any(r[8] for r in rows):
        lines.append(f"### Coherence samples (prompt: *\"{args.sample_prompt}\"*, "
                     f"temp {args.sample_temp}, {args.sample_tokens} new tokens)\n")
        for level, bits, n_wt, blob_kb, loss, tpb, bpb, d_bpb, sample in rows:
            if sample:
                lines.append(f"- **{level}** (bpb {bpb:.3f}): {sample.strip()}")
        lines.append("")
    lines.append("Weight blob includes one fp16 group-scale per group of "
                 f"{args.group_size} (≈ {16/args.group_size:.2f} bits/weight overhead). "
                 "Ternary uses BitNet b1.58 absmean scaling; int8/int4/int2 use symmetric absmax.\n")
    lines.append("**Reading it:** the point where Δ bpb turns from negligible to large marks the "
                 "post-training bit-width floor. Sub-int4 typically needs quantization-aware "
                 "training (BitNet) to recover — a naive PTQ collapse here motivates QAT as the "
                 "next step, not a dead end.\n")
    path = os.path.normpath(os.path.join(os.getcwd(), args.out_md))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
