"""
Generate completions for the BLINDED FRONTIER RE-GRADE.

Every prior coherence verdict in this project (the Coherent/Marginal/Broken tiers, "coherence
onset ~185K", "smallest coherent artifact") came from panels that were leniently calibrated and,
critically, UNBLINDED — `eval/split_batches.py` hands the model label to the judge. This regenerates
the whole frontier so it can be graded blind, in one panel, against the published TinyStories
checkpoints as external calibration anchors.

Covers the floor ladder (75K -> 279K, the range where "onset" was claimed), the deployment
artifacts, the best sub-1M model, and both paper baselines.

Sampling matched across families: temperature 0.8, top-k 200, seed 1337+i, truncated to the same
300 characters (the baselines' 50K vocab emits ~4 bytes/token against this project's ~2, so equal
token budgets would not be equal text). Deterministic, so it reproduces the overlapping subset of
`gen_baseline.py` exactly.

    python eval/gen_frontier_blind.py     # -> eval/completions_frontier_blind.json
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

# (label, kind, spec, tokenizer)
MODELS = [
    # --- floor ladder: the range where "coherence onset ~185K" was claimed ---
    ("d48/2L",              "ours", "out_2L_d48",              TOK512),
    ("d32/5L",              "ours", "out_width_d32",           TOK512),
    ("d64/2L",              "ours", "out_2L_d64",              TOK512),
    ("d48/5L",              "ours", "out_width_d48",           TOK512),
    ("d80/2L (claimed floor)", "ours", "out_2L_d80",           TOK512),
    ("d96/2L",              "ours", "out_iso_d96L2",           TOK512),
    ("A d64/5L fp32",       "ours", "out",                     TOK512),
    # --- deployment artifacts / frontier ---
    ("A + no-fallback tok", "ours", "out_tok_512dep",          TOKDEP),
    ("QAT int4 dep",        "ours", "out_qat_int4_dep",        TOKDEP),
    ("QAT ternary d96 dep", "ours", "out_qat_ternary_d96_dep", TOKDEP),
    ("Width d128 (best sub-1M)", "ours", "out_width_d128",     TOK512),
    # --- external calibration anchors ---
    ("TinyStories-1M (paper)", "hf", "roneneldan/TinyStories-1M", None),
    ("TinyStories-3M (paper)", "hf", "roneneldan/TinyStories-3M", None),
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
    out_path = os.path.join(HERE, "completions_frontier_blind.json")
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

    print(f"\nDone. {len(out)} completions, {len(MODELS)} models -> {out_path}")


if __name__ == "__main__":
    main()
