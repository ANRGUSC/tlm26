"""
Generate TinyStories-style completions from the existing sub-1M DENSE checkpoints, on CPU
(the GPU is busy training). Saves incrementally to eval/completions_sub1m.json after each
model so partial results survive an interruption.

    python eval/gen_sub1m_grade.py            # all candidates
    python eval/gen_sub1m_grade.py out_H      # a single out_dir (timing probe)
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import Tokenizer
from model import ModelArgs, Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COMPLETION_CHARS = 300
MAX_NEW_TOKENS = 220
TEMPERATURE = 0.8
TOP_K = 200
SEED = 1337
N_BEGINNINGS = 20

# (label, out_dir) — dense sub-1M candidates, best bits/byte first
MODELS = [
    ("Width d128 (dim128/5L)", "out_width_d128"),  # ~989K
    ("H (dim192/2L)",          "out_H"),            # ~910K
    ("G (dim128/3L)",          "out_G"),            # ~619K
    ("D (dim96/5L)",           "out_D"),            # ~557K
    ("E (dim64/8L)",           "out_E"),            # ~427K
    ("Depth L6 (dim64/6L)",    "out_depth_L6"),     # ~329K
]


def load_model(out_dir):
    ckpt = torch.load(os.path.join(ROOT, out_dir, "ckpt.pt"), map_location="cpu")
    model = Transformer(ModelArgs(**ckpt["model_args"]))
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.eval()
    return model, sum(p.numel() for p in model.parameters())


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    models = [m for m in MODELS if only is None or m[1] == only]
    beginnings = json.load(open(os.path.join(HERE, "beginnings.json"), encoding="utf-8"))[:N_BEGINNINGS]
    enc = Tokenizer(os.path.join(ROOT, "data", "tok512.model"))
    out_path = os.path.join(HERE, "completions_sub1m.json")
    out = []
    if os.path.exists(out_path) and only is None:
        pass  # fresh run overwrites
    for label, out_dir in models:
        model, nparams = load_model(out_dir)
        tag = f"{label} ({round(nparams / 1000)}K)"
        print(f"[{tag}] {nparams:,} params", flush=True)
        t0 = time.time()
        for i, begin in enumerate(beginnings):
            torch.manual_seed(SEED + i)
            ids = enc.encode(begin, bos=True, eos=False)
            x = torch.tensor(ids, dtype=torch.long)[None, ...]
            with torch.no_grad():
                y = model.generate(x, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
            full = enc.decode(y[0].tolist())
            completion = full[len(begin):][:COMPLETION_CHARS].strip()
            out.append({"model": tag, "params": nparams, "prompt": begin, "completion": completion})
            print(f"  [{i+1}/{len(beginnings)}] {time.time()-t0:5.1f}s", flush=True)
        del model
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  saved {len(out)} completions ({time.time()-t0:.0f}s for this model)", flush=True)
    print(f"\nDone. {len(out)} completions -> {out_path}")


if __name__ == "__main__":
    main()
