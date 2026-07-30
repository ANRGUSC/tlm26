"""
Generate completions for the coherence-FLOOR ladder, TinyStories-style, to map where
coherent English emerges below ~280K params. Outputs eval/completions_floor.json.

Ladder spans ~75K -> ~280K across two shape families (2-layer width sweep and the
5-layer width sweep) so the floor can be read against both size and shape.

    python eval/gen_floor.py
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

# (label, out_dir, vocab_size, looped) — ordered by parameter count
MODELS = [
    ("d48/2L (75K)",  "out_2L_d48",    512, False),
    ("d32/5L (78K)",  "out_width_d32", 512, False),
    ("d64/2L (123K)", "out_2L_d64",    512, False),
    ("d48/5L (152K)", "out_width_d48", 512, False),
    ("d80/2L (182K)", "out_2L_d80",    512, False),
    ("d96/2L (252K)", "out_iso_d96L2", 512, False),
    ("A d64/5L (279K)", "out",         512, False),
]


def load_model(out_dir, looped):
    if looped:
        from model_looped import LoopedModelArgs as MA, LoopedTransformer as TF
    else:
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
            out.append({"model": label, "params": nparams, "prompt": begin, "completion": completion})
        del model
        torch.cuda.empty_cache()
    with open(os.path.join(HERE, "completions_floor.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(out)} completions to eval/completions_floor.json")


if __name__ == "__main__":
    main()
