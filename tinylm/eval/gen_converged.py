"""
Coherence completions for the converged models: the iso-param shape pair (does the
width>depth inversion also show in judged coherence?), the converged deployable QAT int4,
and the paper baselines as calibration anchors.

Matched sampling: temperature 0.8, top-k 200, seed 1337+i, truncated to 300 characters.

    python eval/gen_converged.py     # -> eval/completions_converged.json
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
CC, MNT, TEMP, TOPK, SEED, N = 300, 220, 0.8, 200, 1337, 20
TOK512 = "data/tok512.model"
TOKDEP = "data/tok512dep.model"

MODELS = [
    ("d96/2L wide-shallow @30k",  "ours", "out_iso_d96L2_long",     TOK512),
    ("d48/10L deep-narrow @30k",  "ours", "out_iso_d48L10_long",    TOK512),
    ("QAT int4 dep @30k (deploy)", "ours", "out_qat_int4_dep_long",  TOKDEP),
    ("TinyStories-1M (paper)",     "hf",   "roneneldan/TinyStories-1M", None),
    ("TinyStories-3M (paper)",     "hf",   "roneneldan/TinyStories-3M", None),
]


def load_ours(out_dir):
    ck = torch.load(os.path.join(ROOT, out_dir, "ckpt.pt"), map_location="cpu")
    m = Transformer(ModelArgs(**ck["model_args"]))
    sd = ck["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    m.load_state_dict(sd)
    m.eval()
    return m, sum(p.numel() for p in m.parameters())


def main():
    beg = json.load(open(os.path.join(HERE, "beginnings.json"), encoding="utf-8"))[:N]
    out_path = os.path.join(HERE, "completions_converged.json")
    out = []
    for label, kind, spec, tok in MODELS:
        t0 = time.time()
        if kind == "hf":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokz = AutoTokenizer.from_pretrained(spec)
            model = AutoModelForCausalLM.from_pretrained(spec).eval()
            nparams = sum(p.numel() for p in model.parameters())
            n_embed = model.get_input_embeddings().weight.numel()
        else:
            enc = Tokenizer(os.path.join(ROOT, tok))
            model, nparams = load_ours(spec)
            n_embed = model.tok_embeddings.weight.numel()
        tag = f"{label} ({round(nparams/1000)}K)"
        print(f"[{tag}] {nparams:,}", flush=True)
        for i, b in enumerate(beg):
            torch.manual_seed(SEED + i)
            if kind == "hf":
                ids = tokz.encode(b, return_tensors="pt")
                with torch.no_grad():
                    y = model.generate(ids, max_new_tokens=MNT, do_sample=True,
                                       temperature=TEMP, top_k=TOPK, pad_token_id=tokz.eos_token_id)
                full = tokz.decode(y[0], skip_special_tokens=True)
            else:
                ids = enc.encode(b, bos=True, eos=False)
                x = torch.tensor(ids, dtype=torch.long)[None, ...]
                with torch.no_grad():
                    y = model.generate(x, MNT, temperature=TEMP, top_k=TOPK)
                full = enc.decode(y[0].tolist())
            out.append({"model": tag, "params": nparams, "non_embed": nparams - n_embed,
                        "prompt": b, "completion": full[len(b):][:CC].strip()})
        del model
        json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"  saved {len(out)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"Done -> {out_path}")


if __name__ == "__main__":
    main()
