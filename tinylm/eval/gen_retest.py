"""
Independent retest generation for Fable's Section 13 (GPT-4 data) and Section 14 (QAT) models,
plus the fp32 Model A anchor. CPU, same protocol as eval/gen_sub1m_grade.py (temp 0.8, top_k 200,
seed 1337, first 20 beginnings). QAT checkpoints carry qat_level in model_args, so model.py runs
the fake-quantized forward automatically. Saves eval/completions_retest.json.
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

# (label, out_dir)
MODELS = [
    ("A fp32 (dim64/5L, reference)",  "out"),
    ("QAT int4 (dim64/5L)",           "out_qat_int4"),
    ("QAT ternary (dim64/5L)",        "out_qat_ternary"),
    ("QAT ternary d96 (dim96/5L)",    "out_qat_ternary_d96"),
    ("GPT4-data A (dim64/5L)",        "out_gpt4_A"),
    ("GPT4-data 150k (dim48/5L)",     "out_gpt4_150k"),
    ("GPT4-data F (dim48/4L)",        "out_gpt4_F"),
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
    qat = ckpt["model_args"].get("qat_level", "") or "fp32"
    return model, sum(p.numel() for p in model.parameters()), qat


def main():
    beginnings = json.load(open(os.path.join(HERE, "beginnings.json"), encoding="utf-8"))[:N_BEGINNINGS]
    enc = Tokenizer(os.path.join(ROOT, "data", "tok512.model"))
    out_path = os.path.join(HERE, "completions_retest.json")
    out = []
    for label, out_dir in MODELS:
        model, nparams, qat = load_model(out_dir)
        tag = f"{label} ({round(nparams/1000)}K, {qat})"
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
            out.append({"model": tag, "params": nparams, "qat": qat, "prompt": begin, "completion": completion})
        del model
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  saved {len(out)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"\nDone -> {out_path}")


if __name__ == "__main__":
    main()
