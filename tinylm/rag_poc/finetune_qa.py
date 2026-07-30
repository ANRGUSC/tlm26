# Instruction-tune a checkpoint on QA pairs (answer tokens supervised) -> new checkpoint dir.
import argparse
import json
import math
import os
import random
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from model import ModelArgs, Transformer
from tokenizer import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt_dir", type=str, required=True, help="source checkpoint (read-only)")
ap.add_argument("--tok", type=str, default="data/tok512.model")
ap.add_argument("--data", type=str, required=True)
ap.add_argument("--out_dir", type=str, required=True)
ap.add_argument("--steps", type=int, default=1500)
ap.add_argument("--batch_size", type=int, default=64)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--warmup", type=int, default=75)
ap.add_argument("--weight_decay", type=float, default=0.01)
ap.add_argument("--val_frac", type=float, default=0.02)
ap.add_argument("--eval_every", type=int, default=250)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(args.seed)
rng = random.Random(args.seed)

enc = Tokenizer(os.path.join(ROOT, args.tok))

ck = torch.load(os.path.join(ROOT, args.ckpt_dir, "ckpt.pt"), map_location="cpu")
margs = ck["model_args"]
model = Transformer(ModelArgs(**margs))
sd = ck["model"]
for k in list(sd.keys()):
    if k.startswith("_orig_mod."):
        sd[k[len("_orig_mod."):]] = sd.pop(k)
model.load_state_dict(sd)
model.to(DEV).train()
max_seq = margs["max_seq_len"]

# ---- tokenize all examples once; mask everything before the answer ----
def encode_example(ex):
    prompt = f"{ex['chunk']}\nQuestion: {ex['question']}\nAnswer:"
    full = f"{prompt} {ex['answer']}"
    p_ids = enc.encode(prompt, bos=True, eos=False)
    f_ids = enc.encode(full, bos=True, eos=True)
    if len(f_ids) > max_seq or f_ids[: len(p_ids)] != p_ids:
        # drop the rare example where sentencepiece merges across the prompt boundary
        return None
    x = f_ids[:-1]
    y = [-1] * len(x)
    for i in range(len(p_ids) - 1, len(f_ids) - 1):
        y[i] = f_ids[i + 1]           # answer tokens + eos supervised, prompt masked
    return x, y

examples = []
with open(os.path.join(ROOT, args.data), encoding="utf-8") as f:
    for line in f:
        t = encode_example(json.loads(line))
        if t is not None:
            examples.append(t)
rng.shuffle(examples)
n_val = max(1, int(len(examples) * args.val_frac))
val_ex, train_ex = examples[:n_val], examples[n_val:]
print(f"examples: {len(train_ex)} train / {len(val_ex)} val (dropped none-fit)")

def make_batch(pool, idxs):
    xs = [pool[i] for i in idxs]
    width = max(len(x) for x, _ in xs)
    X = torch.zeros(len(xs), width, dtype=torch.long)
    Y = torch.full((len(xs), width), -1, dtype=torch.long)
    for j, (x, y) in enumerate(xs):
        X[j, : len(x)] = torch.tensor(x, dtype=torch.long)
        Y[j, : len(y)] = torch.tensor(y, dtype=torch.long)
    return X.to(DEV), Y.to(DEV)

opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                        betas=(0.9, 0.95))

def lr_at(it):
    if it < args.warmup:
        return args.lr * it / args.warmup
    t = (it - args.warmup) / max(1, args.steps - args.warmup)
    return args.lr * 0.5 * (1.0 + math.cos(math.pi * t))

ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if DEV == "cuda" \
    else torch.autocast(device_type="cpu", enabled=False)

@torch.no_grad()
def val_loss():
    model.eval()
    tot, n = 0.0, 0
    for b in range(0, len(val_ex), args.batch_size):
        X, Y = make_batch(val_ex, range(b, min(b + args.batch_size, len(val_ex))))
        with ctx:
            model(X, Y)
        tot += model.last_loss.item() * X.size(0)
        n += X.size(0)
    model.train()
    return tot / n

order = list(range(len(train_ex)))
rng.shuffle(order)
pos = 0
for it in range(1, args.steps + 1):
    if pos + args.batch_size > len(order):
        rng.shuffle(order)
        pos = 0
    idxs = order[pos: pos + args.batch_size]
    pos += args.batch_size
    for g in opt.param_groups:
        g["lr"] = lr_at(it)
    X, Y = make_batch(train_ex, idxs)
    with ctx:
        model(X, Y)
        loss = model.last_loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    if it % 50 == 0 or it == 1:
        print(f"{it} | loss {loss.item():.4f} | lr {lr_at(it):.2e}", flush=True)
    if it % args.eval_every == 0 or it == args.steps:
        print(f"step {it}: val loss {val_loss():.4f}", flush=True)

out_dir = os.path.join(ROOT, args.out_dir)
os.makedirs(out_dir, exist_ok=True)
torch.save({"model": model.state_dict(), "model_args": margs, "iter_num": args.steps,
            "config": {"finetuned_from": args.ckpt_dir, "data": args.data,
                       "steps": args.steps, "lr": args.lr}},
           os.path.join(out_dir, "ckpt.pt"))
print(f"saved {out_dir}/ckpt.pt (source checkpoint untouched)")
