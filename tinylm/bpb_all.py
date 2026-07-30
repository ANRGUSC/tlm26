"""Batch bits-per-byte over every trained checkpoint; prints a CSV summary."""
import json
import math
import os
from functools import partial

import numpy as np
import torch

from tinystories import Task

# (label, out_dir, vocab_size, looped)
MODELS = [
    ("J", "out_J", 512, False), ("H", "out_H", 512, False),
    ("G", "out_G", 512, False), ("D", "out_D", 512, False),
    ("I", "out_I", 512, False), ("E", "out_E", 512, False),
    ("Loop 2x3", "out_loop_2x3", 512, True), ("A", "out", 512, False),
    ("Distilled", "out_distill", 512, False), ("Loop 1x5", "out_loop_1x5", 512, True),
    ("Loop 2x3+inject", "out_loop_2x3_inject", 512, True), ("F", "out_F", 512, False),
    ("B", "out_B", 1024, False), ("C", "out_C", 2048, False),
]
EVAL_ITERS = 200
DATA = "data"
device = "cuda" if torch.cuda.is_available() else "cpu"


def tokens_per_byte(vocab_size):
    n_tok = np.memmap(os.path.join(DATA, f"tok{vocab_size}", "data00.bin"),
                      dtype=np.uint16, mode="r").size
    with open(os.path.join(DATA, "TinyStories_all_data", "data00.json"), encoding="utf-8") as f:
        stories = json.load(f)
    n_bytes = sum(len(ex["story"].strip().encode("utf-8")) for ex in stories)
    return n_tok / n_bytes


def eval_one(out_dir, vocab_size, looped):
    if looped:
        from model_looped import ModelArgs, Transformer
    else:
        from model import ModelArgs, Transformer
    ckpt = torch.load(os.path.join(out_dir, "ckpt.pt"), map_location=device)
    margs = ckpt["model_args"]
    model = Transformer(ModelArgs(**margs))
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.to(device).eval()
    it = partial(Task.iter_batches, batch_size=128, max_seq_len=margs["max_seq_len"],
                 vocab_size=vocab_size, vocab_source="custom", device=device, num_workers=0)
    bi = it(split="val")
    losses = torch.zeros(EVAL_ITERS)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if device == "cuda" \
        else torch.autocast(device_type="cpu", enabled=False)
    with torch.no_grad():
        for k in range(EVAL_ITERS):
            X, Y = next(bi)
            with ctx:
                model(X, Y)
                losses[k] = model.last_loss.item()
    loss = losses.mean().item()
    tpb = tokens_per_byte(vocab_size)
    bpb = loss * tpb / math.log(2)
    nparams = sum(p.numel() for p in model.parameters())
    return nparams, loss, tpb, bpb


print("label,params,vocab,val_loss,tokens_per_byte,bits_per_byte")
for label, d, vs, looped in MODELS:
    try:
        p, loss, tpb, bpb = eval_one(d, vs, looped)
        print(f"{label},{p},{vs},{loss:.4f},{tpb:.4f},{bpb:.4f}", flush=True)
    except Exception as e:
        print(f"{label},ERROR,{vs},,,{type(e).__name__}: {e}", flush=True)
