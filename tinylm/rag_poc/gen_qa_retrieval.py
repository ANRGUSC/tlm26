# Retrieval-format QA data: same questions as gen_qa_data.py, but the context is what the
# rag_demo.py retriever actually returns (top-k sentence chunks), so fine-tuning matches
# deployment. The gold sentence is force-included, giving gold + retrieved distractors —
# the signal that teaches the model to copy the RIGHT entity.
import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from tokenizer import Tokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--tok", type=str, default="data/tok512.model")
ap.add_argument("--out_prefix", type=str, default="rag_poc/qaret512")
ap.add_argument("--train_shards", type=int, nargs="+", default=[1, 2])
ap.add_argument("--n_train", type=int, default=20000)
ap.add_argument("--n_eval", type=int, default=600)
ap.add_argument("--top_k", type=int, default=2, help="retrieved chunks (matches rag_demo default)")
ap.add_argument("--max_total_tokens", type=int, default=220)
ap.add_argument("--rare_max", type=int, default=5)
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

rng = random.Random(args.seed)
enc = Tokenizer(os.path.join(ROOT, args.tok))

NAME_RE = re.compile(r"\b([a-z]+) named ([A-Z][a-z]+)")
SENT_RE = re.compile(r"(?<=[.!?\"])\s+")

ROLE_OK = {
    "boy", "girl", "man", "woman", "dog", "cat", "bird", "bunny", "rabbit", "bear",
    "fish", "duck", "frog", "mouse", "monkey", "elephant", "lion", "tiger", "fox",
    "squirrel", "turtle", "puppy", "kitten", "horse", "cow", "pig", "sheep", "chicken",
    "dragon", "dinosaur", "robot", "princess", "prince", "king", "queen", "baby",
    "brother", "sister", "friend", "butterfly", "bee", "ant", "owl", "snake",
}
GENERIC_Q = ["What was the character's name?", "Who was the story about?",
             "What was the name of the main character?"]
ROLE_Q = ["What was the {role}'s name?", "What was the name of the {role}?"]
EVAL_ONLY_Q = "Who is this story about?"

# --- retriever: identical scoring to rag_poc/rag_demo.py (ports to C on the ESP32) ---
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


def sentences(text):
    return [s for s in SENT_RE.split(text.replace("\n", " ").strip()) if s]


def retrieved_context(sents, chunk_bags, df, N, question, gold_idx, k):
    """Top-k sentence chunks by TF-IDF overlap, gold force-included, joined in doc order."""
    qb = bag(question)
    scored = []
    for i, cb in enumerate(chunk_bags):
        s = sum(cb[w] * math.log(1 + N / df[w]) for w in qb if w in cb)
        scored.append((s, i))
    scored.sort(reverse=True)
    top = [i for s, i in scored[:k] if s > 0]
    if gold_idx not in top:                       # guarantee answerability
        if len(top) >= k:
            top[-1] = gold_idx                    # swap out the weakest distractor
        else:
            top.append(gold_idx)
    idx = sorted(set(top))
    return " ".join(sents[i] for i in idx)


def build_examples(story, questions_from, allow_generic=True):
    sents = sentences(story)
    if not sents:
        return []
    mentions = []
    for i, s in enumerate(sents):
        for m in NAME_RE.finditer(s):
            mentions.append((m.group(1), m.group(2), i))
    if not mentions:
        return []
    by_name = {}
    for role, name, idx in mentions:
        by_name.setdefault(name, (role, idx))     # first "role named Name" is the gold sentence
    names = list(by_name.items())
    story_multi = len(names) >= 2

    chunk_bags = [bag(s) for s in sents]
    df = Counter()
    for b in chunk_bags:
        df.update(set(b))
    N = len(sents)

    out = []
    for name, (role, gold_idx) in names:
        if story_multi:
            if role not in ROLE_OK:
                continue
            q = rng.choice(questions_from["role"]).format(role=role)
        else:
            if role in ROLE_OK and rng.random() < 0.5:
                q = rng.choice(questions_from["role"]).format(role=role)
            elif allow_generic:
                q = rng.choice(questions_from["generic"])
            else:
                continue
        context = retrieved_context(sents, chunk_bags, df, N, q, gold_idx, args.top_k)
        text = f"{context}\nQuestion: {q}\nAnswer: {name}"
        if len(enc.encode(text, bos=True, eos=True)) > args.max_total_tokens:
            continue
        # multi_entity for bucketing = does the RETRIEVED context hold >=2 distinct names
        ctx_names = {m.group(2) for m in NAME_RE.finditer(context)}
        out.append({"chunk": context, "question": q, "answer": name,
                    "multi_entity": len(ctx_names) >= 2, "role": role})
    return out


TRAIN_Q = {"generic": GENERIC_Q, "role": ROLE_Q}

# ---------------- train set ----------------
train = []
name_counts = Counter()
for shard in args.train_shards:
    path = os.path.join(ROOT, "data", "TinyStories_all_data", f"data{shard:02d}.json")
    with open(path, encoding="utf-8") as f:
        stories = json.load(f)
    rng.shuffle(stories)
    for ex in stories:
        for g in build_examples(ex["story"].strip(), TRAIN_Q):
            train.append(g)
            name_counts[g["answer"]] += 1
        if len(train) >= args.n_train:
            break
    del stories
    if len(train) >= args.n_train:
        break
train = train[: args.n_train]

# ---------------- eval set (official held-out split) ----------------
with open(os.path.join(ROOT, "data", "TinyStories-valid.txt"), encoding="utf-8") as f:
    val_stories = [s.strip() for s in f.read().split("<|endoftext|>") if s.strip()]
EVAL_Q = {"generic": GENERIC_Q + [EVAL_ONLY_Q], "role": ROLE_Q}
evals = []
for story in val_stories:
    for g in build_examples(story, EVAL_Q):
        g["rare_name"] = name_counts[g["answer"]] <= args.rare_max
        evals.append(g)
    if len(evals) >= args.n_eval:
        break
evals = evals[: args.n_eval]

os.makedirs(os.path.dirname(os.path.join(ROOT, args.out_prefix)), exist_ok=True)
with open(os.path.join(ROOT, args.out_prefix + "_train.jsonl"), "w", encoding="utf-8") as f:
    for ex in train:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
with open(os.path.join(ROOT, args.out_prefix + "_eval.jsonl"), "w", encoding="utf-8") as f:
    for ex in evals:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

n_multi_t = sum(1 for e in train if e["multi_entity"])
n_multi_e = sum(1 for e in evals if e["multi_entity"])
n_rare = sum(1 for e in evals if e["rare_name"])
print(f"train: {len(train)} examples ({n_multi_t} multi-entity ctx), "
      f"{len(name_counts)} distinct names, top5={name_counts.most_common(5)}")
print(f"eval : {len(evals)} examples ({n_multi_e} multi-entity ctx, {n_rare} rare-name)")
