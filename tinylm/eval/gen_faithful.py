"""
Faithful TinyStories-protocol generation: ~50 prompt beginnings x 10 completions each
at temperature 1.0 (Eldan & Li, 2023). Outputs eval/completions_data_faithful.json.

    python eval/gen_faithful.py
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
TEMPERATURE = 0.8          # 0.8, not TinyStories' 1.0: at 1.0 our 279K models garble and the
                           # per-completion coherence signal floors out (see methods note). 0.8
                           # discriminates model quality at this scale.
TOP_K = 200
N_COMPLETIONS = 10         # TinyStories protocol: 10 completions per prompt
SEED = 1337

# dataset-size ablation ladder (Model A shape, shrinking training data)
MODELS = [
    ("full 1.85B (A)", "out",            512),
    ("37M (1 shard)",  "out_data_s01",   512),
    ("10M",            "out_data_t10m",  512),
    ("3M",             "out_data_t3m",   512),
    ("1M",             "out_data_t1m",   512),
    ("300K",           "out_data_t300k", 512),
]


def load_model(out_dir):
    from model import ModelArgs, Transformer
    ckpt = torch.load(os.path.join(ROOT, out_dir, "ckpt.pt"), map_location="cuda")
    model = Transformer(ModelArgs(**ckpt["model_args"]))
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
    for label, out_dir, vocab_size in MODELS:
        enc = Tokenizer(os.path.join(ROOT, "data", f"tok{vocab_size}.model"))
        model, nparams = load_model(out_dir)
        print(f"[{label}] {nparams:,} params, {len(beginnings)} prompts x {N_COMPLETIONS}", flush=True)
        for i, begin in enumerate(beginnings):
            ids = enc.encode(begin, bos=True, eos=False)
            x0 = torch.tensor(ids, dtype=torch.long, device="cuda")[None, ...]
            for j in range(N_COMPLETIONS):
                torch.manual_seed(SEED + i * 100 + j)
                with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    y = model.generate(x0, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
                full = enc.decode(y[0].tolist())
                completion = full[len(begin):][:COMPLETION_CHARS].strip()
                out.append({"model": label, "params": nparams, "prompt": begin,
                            "sample": j, "completion": completion})
        del model
        torch.cuda.empty_cache()
    with open(os.path.join(HERE, "completions_data_faithful.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(out)} completions to eval/completions_data_faithful.json")


if __name__ == "__main__":
    main()
