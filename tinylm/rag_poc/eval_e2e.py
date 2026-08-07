"""True end-to-end RAG eval on unseen stories: real retrieval (NO gold forcing) -> answer.
Decomposes error into retrieval recall (did the gold sentence get retrieved) vs model EM
given the gold was retrieved. This separates the retriever from the model.

    python rag_poc/eval_e2e.py --ckpt_dir out_chat_d128 --n_stories 400
"""
import argparse
import json
import math
import os
import re
import sys
from collections import Counter

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from model import ModelArgs, Transformer
from tokenizer import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt_dir", type=str, required=True)
ap.add_argument("--tok", type=str, default="data/tok512.model")
ap.add_argument("--story_file", type=str, default="data/TinyStories-valid.txt")
ap.add_argument("--n_stories", type=int, default=400)
ap.add_argument("--top_k", type=int, default=2)
ap.add_argument("--max_new", type=int, default=10)
ap.add_argument("--rare_train", type=str, default="rag_poc/qaret512_train.jsonl")
ap.add_argument("--rare_max", type=int, default=5)
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

name_counts = Counter()
for line in open(os.path.join(ROOT, args.rare_train), encoding="utf-8"):
    name_counts[json.loads(line)["answer"]] += 1

NAME_RE = re.compile(r"\b([a-z]+) named ([A-Z][a-z]+)")
SENT_RE = re.compile(r"(?<=[.!?\"])\s+")
WORD = re.compile(r"[a-z']+")
STOP = {"the", "a", "an", "was", "is", "were", "what", "who", "of", "to", "and", "in",
        "it", "he", "she", "they", "his", "her", "s"}


def norm(w):
    if w.endswith("'s"):
        w = w[:-2]
    if w == "named":
        w = "name"
    return w


def bag(text):
    return Counter(norm(w) for w in WORD.findall(text.lower()) if norm(w) not in STOP)


@torch.no_grad()
def answer(prompt):
    ids = enc.encode(prompt, bos=True, eos=False)
    x = torch.tensor(ids, dtype=torch.long, device=DEV)[None, :]
    y = model.generate(x, max_new_tokens=args.max_new, temperature=0.0)
    new = y[0, len(ids):].tolist()
    if enc.eos_id in new:
        new = new[: new.index(enc.eos_id)]
    return enc.decode(new).split("\n")[0].strip()


stories = [s.strip() for s in open(os.path.join(ROOT, args.story_file), encoding="utf-8")
           .read().split("<|endoftext|>") if s.strip()][: args.n_stories]

tot = hit = recall = hit_given_recall = n_recall = 0
rare_tot = rare_hit = 0
for story in stories:
    sents = [s for s in SENT_RE.split(story.replace("\n", " ").strip()) if s]
    if not sents:
        continue
    chunk_bags = [bag(s) for s in sents]
    df = Counter()
    for b in chunk_bags:
        df.update(set(b))
    N = len(sents)
    # gold: first "<role> named <Name>" per name, with its sentence index
    gold = {}
    for i, s in enumerate(sents):
        for m in NAME_RE.finditer(s):
            gold.setdefault(m.group(2), (m.group(1), i))
    story_multi = len(gold) >= 2
    for name, (role, gidx) in gold.items():
        if story_multi and role not in {"boy", "girl", "man", "woman", "dog", "cat", "bird",
            "bunny", "rabbit", "bear", "fish", "duck", "frog", "mouse", "monkey", "elephant",
            "lion", "tiger", "fox", "squirrel", "turtle", "puppy", "kitten", "horse", "cow",
            "pig", "sheep", "chicken", "dragon", "dinosaur", "robot", "princess", "prince",
            "king", "queen", "baby", "brother", "sister", "friend", "butterfly", "bee", "ant",
            "owl", "snake"}:
            continue
        q = f"What was the {role}'s name?"
        qb = bag(q)
        scored = sorted(((sum(chunk_bags[i][w] * math.log(1 + N / df[w])
                              for w in qb if w in chunk_bags[i]), i) for i in range(N)),
                        reverse=True)
        idx = sorted(i for s, i in scored[:args.top_k] if s > 0) or [0]
        got = gidx in idx
        context = " ".join(sents[i] for i in idx)
        pred = answer(f"{context}\nQuestion: {q}\nAnswer:")
        ok = int(pred == name)
        tot += 1; hit += ok; recall += got
        if got:
            n_recall += 1; hit_given_recall += ok
        if name_counts[name] <= args.rare_max:
            rare_tot += 1; rare_hit += ok

print(f"ckpt={args.ckpt_dir}  stories={len(stories)}  questions={tot}")
print(f"  end-to-end EM      = {hit}/{tot} = {hit/tot:.3f}")
print(f"  retrieval recall   = {recall}/{tot} = {recall/tot:.3f}  (gold sentence in top-{args.top_k})")
print(f"  EM | gold retrieved= {hit_given_recall}/{n_recall} = {hit_given_recall/max(1,n_recall):.3f}")
print(f"  rare-name EM       = {rare_hit}/{rare_tot} = {rare_hit/max(1,rare_tot):.3f}")
