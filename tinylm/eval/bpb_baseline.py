"""
Bits-per-byte for ORIGINAL TinyStories checkpoints (Eldan & Li, 2023) and for this
project's checkpoints, over identical validation text and an identical windowing
protocol, so the two families can be compared head-to-head.

The project's own bits_per_byte.py samples random batches from a pretokenized .bin
stream, which only exists for this project's tokenizers. A HuggingFace GPT-Neo model
has its own tokenizer and no .bin, so this script re-derives bpb from raw text:

    bpb = total_NLL_nats / total_val_bytes / ln(2)

Every predicted token contributes exactly once, so the estimate is deterministic
(no batch sampling) and the denominator is the same UTF-8 byte count for every model.
Context is reset at each window boundary and is identical across models.

Usage:
    python eval/bpb_baseline.py --hf roneneldan/TinyStories-1M --seq 256
    python eval/bpb_baseline.py --out_dir out --tok data/tok512.model --seq 512
"""
import argparse
import json
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--hf", type=str, default=None, help="HuggingFace model id (e.g. roneneldan/TinyStories-1M)")
ap.add_argument("--out_dir", type=str, default=None, help="project ckpt dir (contains ckpt.pt)")
ap.add_argument("--tok", type=str, default="data/tok512.model", help="sentencepiece model for --out_dir")
ap.add_argument("--looped", action="store_true")
ap.add_argument("--seq", type=int, default=256, help="context window used for scoring")
ap.add_argument("--n_stories", type=int, default=2000, help="val stories from shard 0")
ap.add_argument("--text_file", type=str, default=None,
                help="score this <|endoftext|>-separated .txt instead of val shard 0 "
                     "(for the contamination probe against the official held-out split)")
ap.add_argument("--batch_size", type=int, default=32)
ap.add_argument("--device", type=str, default="cuda")
ap.add_argument("--tag", type=str, default=None)
args = ap.parse_args()

VAL_JSON = os.path.join(ROOT, "data", "TinyStories_all_data", "data00.json")


def load_val_stories(n):
    with open(VAL_JSON, encoding="utf-8") as f:
        stories = json.load(f)
    return [ex["story"].strip() for ex in stories[:n]]


def load_text_file(path, n):
    """Official TinyStories .txt splits are stories separated by <|endoftext|>."""
    with open(path, encoding="utf-8") as f:
        raw = f.read()
    stories = [s.strip() for s in raw.split("<|endoftext|>") if s.strip()]
    return stories[:n]


IGNORE = -1  # project model.py hardcodes ignore_index=-1 in its internal loss; match it


def score(stream, forward, seq, device, batch_size):
    """Sum NLL (nats) over every token in `stream` that has a predecessor.

    Windows tile the stream so each token index 1..N-1 is predicted exactly once.
    `forward(X, Y)` must return FULL per-position logits (the project's Transformer
    only returns them when targets are passed).
    """
    x_wins, y_wins = [], []
    for i in range(0, len(stream) - 1, seq):
        x = stream[i:i + seq]
        y = stream[i + 1:i + 1 + len(x)]
        x = x[:len(y)]
        if len(x) == 0:
            continue
        x_wins.append(x)
        y_wins.append(y)

    total_nll, total_tok = 0.0, 0
    for b in range(0, len(x_wins), batch_size):
        xs, ys = x_wins[b:b + batch_size], y_wins[b:b + batch_size]
        width = max(len(x) for x in xs)
        # right-pad the ragged tail window; padded positions are masked out of the sum
        X = torch.zeros(len(xs), width, dtype=torch.long)
        Y = torch.full((len(xs), width), IGNORE, dtype=torch.long)
        for j, (x, y) in enumerate(zip(xs, ys)):
            X[j, :len(x)] = torch.tensor(x, dtype=torch.long)
            Y[j, :len(y)] = torch.tensor(y, dtype=torch.long)
        X, Y = X.to(device), Y.to(device)
        with torch.no_grad():
            logits = forward(X, Y)
            assert logits.shape[1] == X.shape[1], (
                f"expected full logits {X.shape[1]}, got {logits.shape[1]} — "
                "project Transformer returns last-position only unless targets are passed"
            )
            nll = torch.nn.functional.cross_entropy(
                logits.float().view(-1, logits.size(-1)), Y.view(-1),
                ignore_index=IGNORE, reduction="sum",
            )
        total_nll += nll.item()
        total_tok += int((Y != IGNORE).sum().item())
    return total_nll, total_tok


def main():
    device = args.device if torch.cuda.is_available() else "cpu"
    if args.text_file:
        stories = load_text_file(args.text_file, args.n_stories)
        source = os.path.basename(args.text_file)
    else:
        stories = load_val_stories(args.n_stories)
        source = "data00.json (project val shard)"
    total_bytes = sum(len(s.encode("utf-8")) for s in stories)

    if args.hf:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokz = AutoTokenizer.from_pretrained(args.hf)
        model = AutoModelForCausalLM.from_pretrained(args.hf).to(device).eval()
        sep = tokz.eos_token_id
        stream = []
        for s in stories:
            stream.append(sep)                       # story separator, mirrors BOS in project pretok
            stream.extend(tokz.encode(s))
        forward = lambda X, Y: model(input_ids=X).logits
        n_params = sum(p.numel() for p in model.parameters())
        n_embed = model.get_input_embeddings().weight.numel()
        label = args.tag or args.hf
        vocab = tokz.vocab_size
    else:
        from tokenizer import Tokenizer
        if args.looped:
            from model_looped import LoopedModelArgs as ModelArgs, LoopedTransformer as Transformer
        else:
            from model import ModelArgs, Transformer
        enc = Tokenizer(os.path.join(ROOT, args.tok))
        ck = torch.load(os.path.join(ROOT, args.out_dir, "ckpt.pt"), map_location="cpu")
        model = Transformer(ModelArgs(**ck["model_args"]))
        sd = ck["model"]
        for k in list(sd.keys()):
            if k.startswith("_orig_mod."):
                sd[k[len("_orig_mod."):]] = sd.pop(k)
        model.load_state_dict(sd)
        model = model.to(device).eval()
        stream = []
        for s in stories:
            stream.extend(enc.encode(s, bos=True, eos=False))  # mirrors tinystories.py pretok
        # targets MUST be passed: without them forward() returns last-position logits only
        forward = lambda X, Y: model(X, Y)
        n_params = sum(p.numel() for p in model.parameters())
        n_embed = model.tok_embeddings.weight.numel()
        label = args.tag or args.out_dir
        vocab = ck["model_args"]["vocab_size"]

    total_nll, total_tok = score(stream, forward, args.seq, device, args.batch_size)
    bpb = total_nll / total_bytes / math.log(2)

    print(f"model             : {label}")
    print(f"params            : {n_params:,}  (embed {n_embed:,} / non-embed {n_params - n_embed:,})")
    print(f"vocab             : {vocab:,}")
    print(f"eval text         : {source}")
    print(f"val stories       : {len(stories):,}   bytes: {total_bytes:,}")
    print(f"tokens scored     : {total_tok:,}   tokens/byte: {total_tok/total_bytes:.4f}")
    print(f"seq (context)     : {args.seq}")
    print(f"loss (nats/tok)   : {total_nll/total_tok:.4f}")
    print(f">>> BITS/BYTE     : {bpb:.4f}")


if __name__ == "__main__":
    main()
