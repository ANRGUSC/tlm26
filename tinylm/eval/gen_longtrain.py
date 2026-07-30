"""
Generate completions for the TRAINING-BUDGET panel.

The blinded frontier re-grade (`verdicts_frontier_blind.md`) left one confound unresolved:
TinyStories-1M beats this project's d128 while holding FEWER non-embedding parameters (529K vs
923K), which points at training budget rather than architecture. This project's fixed protocol is
5000 steps = 655M tokens, under half an epoch of the 1.85B-token dataset, and d128 ended that run
with val loss BELOW train loss - the signature of a model that has not finished learning.

`out_width_d128_long` is the same architecture trained to convergence (30000 steps, val loss
1.1830 -> 1.0144, flat to -0.0002 over the final 1000 steps). It already beats TinyStories-1M on
bits-per-byte on all three eval sets. This panel asks whether that converts into judged coherence.

The 5000-step d128 is included as the control: same architecture, same data, same tokenizer, only
the budget differs, so the pair isolates training budget. The paper baselines anchor calibration.

    python eval/gen_longtrain.py     # -> eval/completions_longtrain.json
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

TOK512 = "data/tok512.model"
TOKDEP = "data/tok512dep.model"

MODELS = [
    ("d128 converged (30k steps)", "ours", "out_width_d128_long", TOK512),
    ("d128 protocol (5k steps)",   "ours", "out_width_d128",      TOK512),   # budget control
    ("QAT int4 dep (5k steps)",    "ours", "out_qat_int4_dep",    TOKDEP),
    ("A d64/5L fp32 (5k steps)",   "ours", "out",                 TOK512),
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
    beginnings = json.load(open(os.path.join(HERE, "beginnings.json"), encoding="utf-8"))[:N_BEGINNINGS]
    out_path = os.path.join(HERE, "completions_longtrain.json")
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
        print(f"[{tag}] {nparams:,} params", flush=True)

        for i, begin in enumerate(beginnings):
            torch.manual_seed(SEED + i)
            if kind == "hf":
                ids = tokz.encode(begin, return_tensors="pt")
                with torch.no_grad():
                    y = model.generate(
                        ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
                        temperature=TEMPERATURE, top_k=TOP_K, pad_token_id=tokz.eos_token_id,
                    )
                full = tokz.decode(y[0], skip_special_tokens=True)
            else:
                ids = enc.encode(begin, bos=True, eos=False)
                x = torch.tensor(ids, dtype=torch.long)[None, ...]
                with torch.no_grad():
                    y = model.generate(x, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
                full = enc.decode(y[0].tolist())
            out.append({
                "model": tag, "params": nparams, "non_embed": nparams - n_embed,
                "prompt": begin, "completion": full[len(begin):][:COMPLETION_CHARS].strip(),
            })

        del model
        json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"  saved {len(out)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nDone. {len(out)} completions -> {out_path}")


if __name__ == "__main__":
    main()
