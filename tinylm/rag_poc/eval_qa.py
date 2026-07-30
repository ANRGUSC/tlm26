# Exact-match eval of a QA-tuned checkpoint; --no_context control; splits by rare/frequent names.
import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from model import ModelArgs, Transformer
from tokenizer import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt_dir", type=str, required=True)
ap.add_argument("--tok", type=str, default="data/tok512.model")
ap.add_argument("--data", type=str, required=True)
ap.add_argument("--no_context", action="store_true")
ap.add_argument("--max_new", type=int, default=10)
ap.add_argument("--show", type=int, default=0, help="print first N transcripts")
args = ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"
enc = Tokenizer(os.path.join(ROOT, args.tok))

ck = torch.load(os.path.join(ROOT, args.ckpt_dir, "ckpt.pt"), map_location="cpu")
model = Transformer(ModelArgs(**ck["model_args"]))
sd = ck["model"]
for k in list(sd.keys()):
    if k.startswith("_orig_mod."):
        sd[k[len("_orig_mod."):]] = sd.pop(k)
model.load_state_dict(sd)
model.to(DEV).eval()

examples = [json.loads(l) for l in open(os.path.join(ROOT, args.data), encoding="utf-8")]


@torch.no_grad()
def answer(prompt):
    ids = enc.encode(prompt, bos=True, eos=False)
    x = torch.tensor(ids, dtype=torch.long, device=DEV)[None, :]
    y = model.generate(x, max_new_tokens=args.max_new, temperature=0.0)
    new = y[0, len(ids):].tolist()
    if enc.eos_id in new:
        new = new[: new.index(enc.eos_id)]
    text = enc.decode(new)
    return text.split("\n")[0].strip()


buckets = {"all": [0, 0], "frequent": [0, 0], "rare": [0, 0],
           "single": [0, 0], "multi": [0, 0]}
shown = 0
for ex in examples:
    if args.no_context:
        prompt = f"Question: {ex['question']}\nAnswer:"
    else:
        prompt = f"{ex['chunk']}\nQuestion: {ex['question']}\nAnswer:"
    pred = answer(prompt)
    hit = int(pred == ex["answer"])
    for b in ["all", "rare" if ex.get("rare_name") else "frequent",
              "multi" if ex.get("multi_entity") else "single"]:
        buckets[b][0] += hit
        buckets[b][1] += 1
    if shown < args.show:
        print(f"--- Q: {ex['question']}  gold: {ex['answer']}  pred: {pred!r}  "
              f"{'OK' if hit else 'MISS'}")
        shown += 1

print(f"\nckpt={args.ckpt_dir}  data={args.data}  no_context={args.no_context}")
for b, (h, n) in buckets.items():
    if n:
        print(f"  {b:<9} EM = {h}/{n} = {h/n:.3f}")
