"""
Generate completions for the DATASET-SIZE ablation models (Model A shape trained on
shrinking data), TinyStories-style, to grade coherence vs training-data size.
Outputs eval/completions_data.json.

    python eval/gen_data.py
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

# (label, out_dir, vocab_size, looped) — ordered by training-data size
MODELS = [
    ("full 1.85B (A)", "out",            512, False),
    ("37M (1 shard)",  "out_data_s01",   512, False),
    ("10M",            "out_data_t10m",  512, False),
    ("3M",             "out_data_t3m",   512, False),
    ("1M",             "out_data_t1m",   512, False),
    ("300K",           "out_data_t300k", 512, False),
]


def load_model(out_dir, looped):
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
    beginnings = json.load(open(os.path.join(HERE, "beginnings.json"), encoding="utf-8"))
    out = []
    for label, out_dir, vocab_size, looped in MODELS:
        enc = Tokenizer(os.path.join(ROOT, "data", f"tok{vocab_size}.model"))
        model, nparams = load_model(out_dir, looped)
        print(f"[{label}] {nparams:,} params", flush=True)
        for i, begin in enumerate(beginnings):
            torch.manual_seed(SEED + i)
            ids = enc.encode(begin, bos=True, eos=False)
            x = torch.tensor(ids, dtype=torch.long, device="cuda")[None, ...]
            with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                y = model.generate(x, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
            full = enc.decode(y[0].tolist())
            completion = full[len(begin):][:COMPLETION_CHARS].strip()
            out.append({"model": label, "data": label, "prompt": begin, "completion": completion})
        del model
        torch.cuda.empty_cache()
    with open(os.path.join(HERE, "completions_data.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(out)} completions to eval/completions_data.json")


if __name__ == "__main__":
    main()
