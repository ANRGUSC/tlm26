# Pure-CE bits/byte + active-param accounting for MoE and dense checkpoints (apples-to-apples).
import json
import math
import os
import sys
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F

from tinystories import Task


def load_model(ckpt_dir):
    ck = torch.load(os.path.join(ckpt_dir, "ckpt.pt"), map_location="cpu", weights_only=False)
    ma = ck["model_args"]
    if ma.get("n_experts"):
        from model_moe import MoETransformer as T, MoEModelArgs as A
    else:
        from model import Transformer as T, ModelArgs as A
    m = T(A(**ma))
    sd = ck["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    m.load_state_dict(sd)
    return m, ma


@torch.no_grad()
def val_ce(m, vocab, device, iters=200):
    m.to(device).eval()
    it = partial(Task.iter_batches, batch_size=128, max_seq_len=m.params.max_seq_len,
                 vocab_size=vocab, vocab_source="custom", device=device, num_workers=0)
    bi = it(split="val")
    tot = 0.0
    for _ in range(iters):
        X, Y = next(bi)
        logits = m(X, Y)
        tot += F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1).item()
    return tot / iters


def tpb(vocab):
    n = np.memmap(os.path.join("data", f"tok{vocab}", "data00.bin"), dtype=np.uint16, mode="r").size
    with open(os.path.join("data", "TinyStories_all_data", "data00.json"), encoding="utf-8") as f:
        stories = json.load(f)
    nb = sum(len(e["story"].strip().encode("utf-8")) for e in stories)
    return n / nb


def params(m, ma):
    total = sum(p.numel() for p in m.parameters())
    if ma.get("n_experts"):
        one = sum(p.numel() for p in m.layers[0].feed_forward.experts[0].parameters())
        inactive = (ma["n_experts"] - ma["moe_top_k"]) * one * ma["n_layers"]
        return total, total - inactive
    return total, total


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LN2 = math.log(2)
    print(f"{'checkpoint':26s}{'kind':16s}{'CE':>8s}{'bpb':>9s}{'total':>12s}{'active/tok':>12s}")
    for d in sys.argv[1:]:
        m, ma = load_model(d)
        ce = val_ce(m, ma["vocab_size"], device)
        bpb = ce * tpb(ma["vocab_size"]) / LN2
        total, active = params(m, ma)
        kind = f"MoE {ma['n_experts']}e/top{ma['moe_top_k']}" if ma.get("n_experts") else "dense"
        print(f"{d:26s}{kind:16s}{ce:>8.4f}{bpb:>9.4f}{total:>12,}{active:>12,}")
