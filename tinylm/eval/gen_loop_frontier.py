"""
Generate completions for the looping-frontier sweep, TinyStories-style, to judge whether
weight-shared (looped) models speak coherent English below the ~185K dense coherence floor.
Outputs eval/completions_loop_frontier.json.

Five models: the four frontier loop shapes plus the n_kv_heads=1 (MQA) variant of the best.

    python eval/gen_loop_frontier.py
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
    ("Loop 1x8 d64",     "out_fr_loop_1x8_d64",     512, True),
    ("Loop 1x5 d80",     "out_fr_loop_1x5_d80",     512, True),
    ("Loop 1x5 d96",     "out_fr_loop_1x5_d96",     512, True),
    ("Loop 2x3 d96 kv1", "out_fr_loop_2x3_d96_kv1", 512, True),
    ("Loop 2x3 d96",     "out_fr_loop_2x3_d96",     512, True),
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
    # first 20 beginnings — the same set §6's floor ladder was graded on
    beginnings = json.load(open(os.path.join(HERE, "beginnings.json"), encoding="utf-8"))[:20]
    out = []
    for label, out_dir, vocab_size, looped in MODELS:
        enc = Tokenizer(os.path.join(ROOT, "data", f"tok{vocab_size}.model"))
        model, nparams = load_model(out_dir, looped)
        label = f"{label} ({round(nparams / 1000)}K)"
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
    with open(os.path.join(HERE, "completions_loop_frontier.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(out)} completions to eval/completions_loop_frontier.json")


if __name__ == "__main__":
    main()
