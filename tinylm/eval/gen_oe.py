"""Generate completions for the over-encoding experiment: baseline (dep A), OE (bigram-2048),
and the param-matched dense control (dim80), all on tok512dep. Output eval/completions_oe.json."""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenizer import Tokenizer
from model import ModelArgs, Transformer

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
CC, MNT, TEMP, TOPK, SEED, N = 300, 220, 0.8, 200, 1337, 20
MODELS = [
    ("baseline dep A (279K)",       "out_tok_512dep"),
    ("OE bigram-2048 (410K)",       "out_oe_2048_dep"),
    ("dense d80 param-matched (406K)", "out_dense_d80_dep"),
]

def load(out_dir):
    ck = torch.load(os.path.join(ROOT, out_dir, "ckpt.pt"), map_location="cpu")
    m = Transformer(ModelArgs(**ck["model_args"])); sd = ck["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."): sd[k[len("_orig_mod."):]] = sd.pop(k)
    m.load_state_dict(sd); m.eval()
    return m, sum(p.numel() for p in m.parameters())

def main():
    beg = json.load(open(os.path.join(HERE, "beginnings.json"), encoding="utf-8"))[:N]
    enc = Tokenizer(os.path.join(ROOT, "data", "tok512dep.model"))
    out = []; out_path = os.path.join(HERE, "completions_oe.json")
    for label, d in MODELS:
        m, n = load(d); tag = f"{label}"
        print(f"[{tag}] {n:,} params", flush=True); t0 = time.time()
        for i, b in enumerate(beg):
            torch.manual_seed(SEED + i)
            ids = enc.encode(b, bos=True, eos=False)
            x = torch.tensor(ids, dtype=torch.long)[None, ...]
            with torch.no_grad():
                y = m.generate(x, MNT, temperature=TEMP, top_k=TOPK)
            out.append({"model": tag, "params": n, "prompt": b, "completion": enc.decode(y[0].tolist())[len(b):][:CC].strip()})
        del m
        json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"  saved {len(out)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"Done -> {out_path}")

if __name__ == "__main__":
    main()
