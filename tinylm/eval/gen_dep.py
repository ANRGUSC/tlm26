"""Generate completions from the tok512dep-tokenizer models (fp32 reference + QAT int4/ternary-d96
retrained on the no-fallback deployment tokenizer). Decodes with tok512dep.model. Same protocol as
gen_retest.py. Output eval/completions_dep.json."""
import json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import Tokenizer
from model import ModelArgs, Transformer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COMPLETION_CHARS, MAX_NEW_TOKENS, TEMPERATURE, TOP_K, SEED, N = 300, 220, 0.8, 200, 1337, 20

MODELS = [
    ("dep fp32 A (dim64/5L, reference)", "out_tok_512dep"),
    ("dep QAT int4 (dim64/5L)",          "out_qat_int4_dep"),
    ("dep QAT ternary d96 (dim96/5L)",   "out_qat_ternary_d96_dep"),
]

def load_model(out_dir):
    ckpt = torch.load(os.path.join(ROOT, out_dir, "ckpt.pt"), map_location="cpu")
    model = Transformer(ModelArgs(**ckpt["model_args"]))
    sd = ckpt["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    model.load_state_dict(sd); model.eval()
    return model, sum(p.numel() for p in model.parameters()), (ckpt["model_args"].get("qat_level","") or "fp32")

def main():
    beginnings = json.load(open(os.path.join(HERE, "beginnings.json"), encoding="utf-8"))[:N]
    enc = Tokenizer(os.path.join(ROOT, "data", "tok512dep.model"))
    out_path = os.path.join(HERE, "completions_dep.json")
    out = []
    for label, out_dir in MODELS:
        model, nparams, qat = load_model(out_dir)
        tag = f"{label} ({round(nparams/1000)}K, {qat})"
        print(f"[{tag}] {nparams:,} params", flush=True); t0 = time.time()
        for i, begin in enumerate(beginnings):
            torch.manual_seed(SEED + i)
            ids = enc.encode(begin, bos=True, eos=False)
            x = torch.tensor(ids, dtype=torch.long)[None, ...]
            with torch.no_grad():
                y = model.generate(x, MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K)
            completion = enc.decode(y[0].tolist())[len(begin):][:COMPLETION_CHARS].strip()
            out.append({"model": tag, "params": nparams, "qat": qat, "prompt": begin, "completion": completion})
        del model
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  saved {len(out)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"Done -> {out_path}")

if __name__ == "__main__":
    main()
