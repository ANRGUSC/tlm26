"""
Generate story completions for each model on the fixed prompt set, TinyStories-style.

For a fair cross-tokenizer comparison we generate to a fixed number of *characters*
of completion (not tokens) — a 105-vocab char model needs far more tokens to write
the same amount of text as a 512-vocab BPE model.

    python eval/gen_completions.py            # all models in MODELS
Outputs eval/completions.json: [{model, params, prompt, completion}, ...]
"""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COMPLETION_CHARS = 250   # truncate each completion to this many chars (fair across tokenizers)
MAX_NEW_TOKENS = 320     # generate generously, then truncate by chars
TEMPERATURE = 0.8
TOP_K = 200
SEED = 1337

# (label, out_dir, vocab_size, looped)
MODELS = [
    ("char (vocab 105)", "out_char105", 105, False),
    ("A baseline (vocab 512)", "out", 512, False),
    ("J best (vocab 512)", "out_J", 512, False),
]


def load_model(out_dir, looped):
    if looped:
        from model_looped import LoopedModelArgs as MA, LoopedTransformer as TF
    else:
        from model import ModelArgs as MA, Transformer as TF
    ckpt = torch.load(os.path.join(ROOT, out_dir, "ckpt.pt"), map_location="cuda")
    margs = ckpt["model_args"]
    model = TF(MA(**margs))
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd)
    model.to("cuda").eval()
    nparams = sum(p.numel() for p in model.parameters())
    return model, nparams


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
            # completion = text after the beginning, truncated to a fixed char budget
            completion = full[len(begin):][:COMPLETION_CHARS].strip()
            out.append({"model": label, "params": nparams, "prompt": begin, "completion": completion})
        del model
        torch.cuda.empty_cache()
    with open(os.path.join(HERE, "completions.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(out)} completions to eval/completions.json")


if __name__ == "__main__":
    main()
