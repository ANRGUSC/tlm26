"""End-to-end RAG demo on a single unseen text file: retrieve → prompt → answer.

Retrieval is deliberately primitive — lowercase word overlap between the question and
each sentence chunk, weighted by inverse chunk frequency (a ~30-line TF-IDF that ports
directly to C on the ESP32; no embeddings, no index). The retrieved chunk plus the
question is formatted exactly like the fine-tuning data and answered greedily.

    python rag_poc/rag_demo.py --ckpt_dir out_rag_qa_d128 --tok data/tok512.model \
        --story_index 3 --ask "What was the dog's name?"
If --ask is omitted, asks every applicable auto-generated name question for the story.
"""
import argparse
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
ap.add_argument("--story_index", type=int, default=0)
ap.add_argument("--ask", type=str, default=None)
ap.add_argument("--top_k", type=int, default=2, help="retrieved chunks")
ap.add_argument("--max_new", type=int, default=10)
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

stories = [s.strip() for s in open(os.path.join(ROOT, args.story_file), encoding="utf-8")
           .read().split("<|endoftext|>") if s.strip()]
story = stories[args.story_index]

# --- chunk: one sentence per chunk (ports to C trivially) ---
chunks = [c.strip() for c in re.split(r"(?<=[.!?\"])\s+", story.replace("\n", " ")) if c.strip()]

WORD = re.compile(r"[a-z']+")
# "name"/"named" are NOT stopwords here: they are the evidence signal for name questions
STOP = {"the", "a", "an", "was", "is", "were", "what", "who", "of", "to", "and", "in",
        "it", "he", "she", "they", "his", "her", "s"}

def norm(w):
    # strip possessive and fold the name/named inflection so "the bird's name"
    # matches "a bird named Billy" — this is the whole stemmer, and it ports to C
    if w.endswith("'s"):
        w = w[:-2]
    if w == "named":
        w = "name"
    return w


def bag(text):
    return Counter(norm(w) for w in WORD.findall(text.lower())
                   if norm(w) not in STOP)

chunk_bags = [bag(c) for c in chunks]
df = Counter()
for b in chunk_bags:
    df.update(set(b))
N = len(chunks)

def retrieve(question, k):
    qb = bag(question)
    scores = []
    for i, cb in enumerate(chunk_bags):
        s = sum(cb[w] * math.log(1 + N / df[w]) for w in qb if w in cb)
        scores.append((s, i))
    scores.sort(reverse=True)
    idx = sorted(i for s, i in scores[:k] if s > 0) or [0]
    return idx

@torch.no_grad()
def answer(prompt):
    ids = enc.encode(prompt, bos=True, eos=False)
    x = torch.tensor(ids, dtype=torch.long, device=DEV)[None, :]
    y = model.generate(x, max_new_tokens=args.max_new, temperature=0.0)
    new = y[0, len(ids):].tolist()
    if enc.eos_id in new:
        new = new[: new.index(enc.eos_id)]
    return enc.decode(new).split("\n")[0].strip()

def ask(question):
    idx = retrieve(question, args.top_k)
    context = " ".join(chunks[i] for i in idx)
    prompt = f"{context}\nQuestion: {question}\nAnswer:"
    pred = answer(prompt)
    print(f"Q: {question}")
    print(f"  retrieved chunks {idx}: {context!r}")
    print(f"  A: {pred}")
    return pred

print(f"story #{args.story_index} ({len(chunks)} sentence chunks, "
      f"{len(enc.encode(story, bos=False, eos=False))} tokens total):\n{story[:300]}...\n")

if args.ask:
    ask(args.ask)
else:
    for m in re.finditer(r"\b([a-z]+) named ([A-Z][a-z]+)", story):
        role, gold = m.group(1), m.group(2)
        pred = ask(f"What was the {role}'s name?")
        print(f"  gold: {gold}  ->  {'OK' if pred == gold else 'MISS'}\n")
