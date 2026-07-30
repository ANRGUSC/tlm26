# Generate synthetic extractive QA pairs ("<role> named <Name>") from TinyStories for the RAG PoC.
import argparse
import json
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
ap.add_argument("--out_prefix", type=str, default="rag_poc/qa512")
ap.add_argument("--train_shards", type=int, nargs="+", default=[1, 2])
ap.add_argument("--n_train", type=int, default=20000)
ap.add_argument("--n_eval", type=int, default=600)
ap.add_argument("--max_chunk_tokens", type=int, default=160)
ap.add_argument("--max_total_tokens", type=int, default=220, help="chunk+question+answer budget (window is 256)")
ap.add_argument("--rare_max", type=int, default=5, help="eval name is 'rare' if seen <= this many times in train")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

rng = random.Random(args.seed)
enc = Tokenizer(os.path.join(ROOT, args.tok))

NAME_RE = re.compile(r"\b([a-z]+) named ([A-Z][a-z]+)")
SENT_RE = re.compile(r"(?<=[.!?\"])\s+")

# Roles that read naturally in "What was the <role>'s name?"; anything else falls back
# to the generic phrasings. Keeps questions grammatical without a full NP parser.
ROLE_OK = {
    "boy", "girl", "man", "woman", "dog", "cat", "bird", "bunny", "rabbit", "bear",
    "fish", "duck", "frog", "mouse", "monkey", "elephant", "lion", "tiger", "fox",
    "squirrel", "turtle", "puppy", "kitten", "horse", "cow", "pig", "sheep", "chicken",
    "dragon", "dinosaur", "robot", "princess", "prince", "king", "queen", "baby",
    "brother", "sister", "friend", "butterfly", "bee", "ant", "owl", "snake",
}

GENERIC_Q = [
    "What was the character's name?",
    "Who was the story about?",
    "What was the name of the main character?",
]
ROLE_Q = [
    "What was the {role}'s name?",
    "What was the name of the {role}?",
]
# Held out from training entirely; eval-only phrasing tests robustness to rewording.
EVAL_ONLY_Q = "Who is this story about?"


def sentences(text):
    return SENT_RE.split(text.replace("\n", " ").strip())


def build_examples(story, questions_from, allow_generic=True):
    """Return a list of dicts for one story, or [] if unusable."""
    sents = sentences(story)
    mentions = []  # (role, name, sent_idx)
    for i, s in enumerate(sents):
        for m in NAME_RE.finditer(s):
            mentions.append((m.group(1), m.group(2), i))
    if not mentions:
        return []
    # chunk: leading sentences covering the LAST mention we plan to use, token-capped
    by_name = {}
    for role, name, idx in mentions:
        by_name.setdefault(name, (role, idx))
    names = list(by_name.items())
    last_idx = max(idx for _, (_, idx) in names)
    chunk_sents = sents[: last_idx + 1]
    chunk = " ".join(chunk_sents)
    toks = enc.encode(chunk, bos=False, eos=False)
    if len(toks) > args.max_chunk_tokens:
        # drop trailing sentences that are past the last needed mention won't help —
        # instead drop names whose mention lies beyond the token cap, re-chunk
        while len(toks) > args.max_chunk_tokens and len(chunk_sents) > 1:
            chunk_sents = chunk_sents[:-1]
            chunk = " ".join(chunk_sents)
            toks = enc.encode(chunk, bos=False, eos=False)
        kept = len(chunk_sents) - 1
        names = [(n, (r, i)) for n, (r, i) in names if i <= kept]
        if not names:
            return []
    multi = len(names) >= 2
    out = []
    for name, (role, _) in names:
        if multi:
            # role-typed question is the only unambiguous form with several names
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
        text = f"{chunk}\nQuestion: {q}\nAnswer: {name}"
        total = enc.encode(text, bos=True, eos=True)
        if len(total) > args.max_total_tokens:
            continue
        out.append({"chunk": chunk, "question": q, "answer": name,
                    "multi_entity": multi, "role": role})
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
        got = build_examples(ex["story"].strip(), TRAIN_Q)
        for g in got:
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
    got = build_examples(story, EVAL_Q)
    for g in got:
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
print(f"train: {len(train)} examples ({n_multi_t} multi-entity), "
      f"{len(name_counts)} distinct names, top5={name_counts.most_common(5)}")
print(f"eval : {len(evals)} examples ({n_multi_e} multi-entity, {n_rare} rare-name)")
