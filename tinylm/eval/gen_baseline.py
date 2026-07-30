"""
Generate completions for the head-to-head against the original TinyStories checkpoints
(Eldan & Li, 2023), for grading in a SINGLE merged judge panel.

The published GPT-4 grades from the paper are NOT comparable to this project's panel: they
use a different prompt set, judge, and calibration. `verdicts_tokcompare.md` records this
project drawing the opposite conclusion from a separately-calibrated panel. So the baselines
are regenerated here on this project's own beginnings and graded alongside this project's
anchors in one panel.

Anchors span the claim under test: the 187K coherence floor, Model A at 279K, the best
sub-1M model at 989K, and the smallest coherent deployable artifact.

Sampling is matched across families: temperature 0.8, top-k 200, seed 1337+i, and every
completion truncated to the same 300 characters (the baselines' 50K vocab emits ~4 bytes per
token against this project's ~2, so truncating on characters rather than tokens is what makes
the graded text comparable).

    python eval/gen_baseline.py            # -> eval/completions_baseline.json
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

# (label, kind, spec, tokenizer) — kind is "hf" or "ours"
MODELS = [
    ("TinyStories-1M (paper baseline)", "hf",   "roneneldan/TinyStories-1M", None),
    ("TinyStories-3M (paper baseline)", "hf",   "roneneldan/TinyStories-3M", None),
    ("Width d128 (ours, best sub-1M)",  "ours", "out_width_d128",            "data/tok512.model"),
    ("A + no-fallback tok (ours)",      "ours", "out_tok_512dep",            "data/tok512dep.model"),
    ("QAT ternary d96 dep (ours)",      "ours", "out_qat_ternary_d96_dep",   "data/tok512dep.model"),
    ("2L d80 (ours, 187K floor)",       "ours", "out_2L_d80",                "data/tok512.model"),
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
    out_path = os.path.join(HERE, "completions_baseline.json")
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
                        temperature=TEMPERATURE, top_k=TOP_K,
                        pad_token_id=tokz.eos_token_id,
                    )
                full = tokz.decode(y[0], skip_special_tokens=True)
            else:
                ids = enc.encode(begin, bos=True, eos=False)
                x = torch.tensor(ids, dtype=torch.long)[None, ...]
                with torch.no_grad():
                    y = model.generate(x, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
                full = enc.decode(y[0].tolist())
            completion = full[len(begin):][:COMPLETION_CHARS].strip()
            out.append({
                "model": tag, "params": nparams, "non_embed": nparams - n_embed,
                "prompt": begin, "completion": completion,
            })

        del model
        json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"  saved {len(out)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nDone. {len(out)} completions -> {out_path}")


if __name__ == "__main__":
    main()
