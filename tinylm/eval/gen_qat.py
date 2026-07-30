"""
Generate completions for the QAT experiment (quantization-aware training at int4 and
ternary), TinyStories-style. The checkpoints carry qat_level in model_args, so the
forward pass fake-quantizes the weights exactly as during training — the completions
reflect the quantized model, not the latent fp32 weights. Outputs eval/completions_qat.json.

    python eval/gen_qat.py
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

MODELS = [
    ("QAT int4 d64/5L",       "out_qat_int4"),
    ("QAT ternary d64/5L",    "out_qat_ternary"),
    ("QAT ternary d96/5L",    "out_qat_ternary_d96"),
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
        print(f"[{label}] {nparams:,} params (qat_level={model.params.qat_level})", flush=True)
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
    with open(os.path.join(HERE, "completions_qat.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(out)} completions to eval/completions_qat.json")


if __name__ == "__main__":
    main()
