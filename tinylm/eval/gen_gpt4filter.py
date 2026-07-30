"""
Generate completions for the data-quality experiment (GPT-4-only training data),
TinyStories-style. Covers the three filtered models and their original-data twins so
the comparison sits in one file under one protocol. Outputs eval/completions_gpt4filter.json.

    python eval/gen_gpt4filter.py
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COMPLETION_CHARS = 250
MAX_NEW_TOKENS = 320
TEMPERATURE = 0.8
TOP_K = 200
SEED = 1337

# (label, out_dir) — original twin followed by its GPT-4-data counterpart
MODELS = [
    ("F d48/4L original",  "out_F"),
    ("F d48/4L GPT-4 data", "out_gpt4_F"),
    ("d48/5L original",    "out_width_d48"),
    ("d48/5L GPT-4 data",  "out_gpt4_150k"),
    ("A d64/5L original",  "out"),
    ("A d64/5L GPT-4 data", "out_gpt4_A"),
]


def load_model(out_dir):
    from model import ModelArgs as MA, Transformer as TF
    ckpt = torch.load(os.path.join(ROOT, out_dir, "ckpt.pt"), map_location="cuda")
    model = TF(MA(**ckpt["model_args"]))
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.to("cuda").eval()
    return model, sum(p.numel() for p in model.parameters())


def main():
    # first 20 beginnings — the same set the Section 6 floor ladder was graded on
    beginnings = json.load(open(os.path.join(HERE, "beginnings.json"), encoding="utf-8"))[:20]
    enc = Tokenizer(os.path.join(ROOT, "data", "tok512.model"))
    out = []
    for label, out_dir in MODELS:
        model, nparams = load_model(out_dir)
        print(f"[{label}] {nparams:,} params", flush=True)
        for i, begin in enumerate(beginnings):
            torch.manual_seed(SEED + i)
            ids = enc.encode(begin, bos=True, eos=False)
            x = torch.tensor(ids, dtype=torch.long, device="cuda")[None, ...]
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                y = model.generate(x, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
            full = enc.decode(y[0].tolist())
            completion = full[len(begin):][:COMPLETION_CHARS].strip()
            out.append({"model": label, "params": nparams, "prompt": begin, "completion": completion})
        del model
        torch.cuda.empty_cache()
    with open(os.path.join(HERE, "completions_gpt4filter.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(out)} completions to eval/completions_gpt4filter.json")


if __name__ == "__main__":
    main()
