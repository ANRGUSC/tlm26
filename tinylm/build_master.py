# One row per trained checkpoint -> ../reports/master_data.csv. bits/byte = val_loss * tok/byte / ln2.
import csv
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT_CSV = os.path.normpath(os.path.join(HERE, "..", "reports", "master_data.csv"))
LN2 = math.log(2)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


_TPB = {}  # tokens/byte per tokenizer, cached


def val_bytes():
    if "_b" not in _TPB:
        with open(os.path.join(DATA, "TinyStories_all_data", "data00.json"), encoding="utf-8") as f:
            stories = json.load(f)
        _TPB["_b"] = sum(len(ex["story"].strip().encode("utf-8")) for ex in stories)
    return _TPB["_b"]


def tokens_per_byte(tokdir):
    if tokdir not in _TPB:
        binp = os.path.join(DATA, tokdir, "data00.bin")
        ntok = np.memmap(binp, dtype=np.uint16, mode="r").size
        _TPB[tokdir] = ntok / val_bytes()
    return _TPB[tokdir]


def tokdir_for(name, vocab):  # map checkpoint dir -> its tokenizer bin dir (segment-based)
    segs = name.split("_")
    if "512dep" in segs:
        return "tok512dep"
    if "512nf" in segs:
        return "tok512nf"
    if "512uni" in segs:
        return "tok512uni"
    if name.startswith("out_gpt4"):
        return "tok512_gpt4"
    if vocab == 512 and "dep" in segs:            # dep/no-fallback tokenizer, vocab 512
        return "tok512dep"
    d = f"tok{vocab}"
    return d if os.path.isdir(os.path.join(DATA, d)) else None


EXP_OVERRIDE = {"out": "size_ladder", "out_B": "vocab", "out_C": "vocab"}


def experiment_for(name):
    if name in EXP_OVERRIDE:
        return EXP_OVERRIDE[name]
    if len(name) == 5 and name[4] in "DEFGHIJ":        # out_D .. out_J
        return "size_ladder"
    rules = [
        ("out_B", "vocab"), ("out_C", "vocab"),            # incl _long retrains
        ("out_width", "width"), ("out_depth", "depth"),
        ("out_2L", "shape"), ("out_iso", "shape"),
        ("out_data_s", "data_shards"), ("out_data_t", "data_tokens"),
        ("out_tok", "tokenizer"), ("out_char", "tokenizer"),
        ("out_qat", "quant"), ("out_dense", "quant"), ("out_oe", "quant"),
        ("out_fr_loop", "looped"), ("out_loop", "looped"),
        ("out_gpt4", "data_quality"), ("out_rag", "rag"), ("out_distill", "distill"),
    ]
    for pre, tag in rules:
        if name.startswith(pre):
            return tag
    return "other"


def is_looped(name):
    return name.startswith("out_loop") or name.startswith("out_fr_loop")


LABEL = {"out": "A", "out_B": "B", "out_C": "C", "out_D": "D", "out_E": "E",
         "out_F": "F", "out_G": "G", "out_H": "H", "out_I": "I", "out_J": "J"}


def label_for(name):
    if name in LABEL:
        return LABEL[name]
    return name[4:] if name.startswith("out_") else name


def count_params(margs, looped):
    if looped:
        from model_looped import LoopedModelArgs as ModelArgs, LoopedTransformer as Transformer
    else:
        from model import ModelArgs, Transformer
    m = Transformer(ModelArgs(**margs))
    return sum(p.numel() for p in m.parameters())


def process(name):
    ckpt = torch.load(os.path.join(HERE, name, "ckpt.pt"), map_location="cpu", weights_only=False)
    ma = ckpt["model_args"]
    vocab = int(ma["vocab_size"])
    dim = int(ma["dim"])
    steps = int(ckpt.get("iter_num", 0))
    looped = is_looped(name)
    params = count_params(ma, looped)
    non_embed = params - vocab * dim                    # tok_embeddings tied to output head
    td = tokdir_for(name, vocab)
    qat = ma.get("qat_level") or ""

    bv = ckpt.get("best_val_loss")
    note = ""
    if bv is None:                                      # fine-tunes (RAG) don't log best_val_loss
        val_loss = tpb = bpb = None
        note = "no best_val_loss (fine-tune); bpb not comparable"
    else:
        val_loss = float(bv)
        tpb = tokens_per_byte(td) if td else float("nan")
        bpb = val_loss * tpb / LN2

    def r4(x):
        return round(x, 4) if x is not None else ""

    return {
        "label": label_for(name), "experiment": experiment_for(name),
        "dim": dim, "n_layers": int(ma["n_layers"]), "n_heads": int(ma["n_heads"]),
        "n_kv_heads": ma.get("n_kv_heads") if ma.get("n_kv_heads") is not None else ma["n_heads"],
        "vocab": vocab, "tokenizer": td or "", "quant": qat,
        "steps": steps, "converged": int(steps >= 30000),
        "params_total": params, "params_non_embed": non_embed,
        "val_loss": r4(val_loss), "tokens_per_byte": r4(tpb),
        "bits_per_byte": r4(bpb), "coherence": "",
        "checkpoint_dir": name, "error": note,
    }


FIELDS = ["label", "experiment", "dim", "n_layers", "n_heads", "n_kv_heads", "vocab",
          "tokenizer", "quant", "steps", "converged", "params_total", "params_non_embed",
          "val_loss", "tokens_per_byte", "bits_per_byte", "coherence", "checkpoint_dir", "error"]

# External released baselines -- context anchors, excluded from the fit.
BASELINES = [
    dict(label="TinyStories-1M", experiment="baseline", vocab=50257, tokenizer="GPT-Neo-50k",
         params_total=3745984, params_non_embed=529536, bits_per_byte=0.8004, coherence=5.19,
         checkpoint_dir="roneneldan/TinyStories-1M", error="continuous data00 (bpb_baseline.py)"),
    dict(label="TinyStories-3M", experiment="baseline", vocab=50257, tokenizer="GPT-Neo-50k",
         params_total=8278400, params_non_embed=1845504, bits_per_byte=0.6501, coherence=6.26,
         checkpoint_dir="roneneldan/TinyStories-3M", error="continuous data00 (bpb_baseline.py)"),
    dict(label="TinyStories-656K", experiment="baseline", vocab=2048, tokenizer="custom-2048",
         params_total=656000, params_non_embed=394000, bits_per_byte=0.969, coherence="",
         checkpoint_dir="raincandy-u/TinyStories-656K", error="native per-story data00 (bpb_native.py)"),
]


def main():
    dirs = sorted(d for d in os.listdir(HERE)
                  if d.startswith("out") and os.path.isfile(os.path.join(HERE, d, "ckpt.pt")))
    rows = []
    for name in dirs:
        try:
            r = process(name)
            log(f"OK  {name:26s} p={r['params_total']:>8d} steps={r['steps']:>5d} "
                f"tok={r['tokenizer']:10s} bpb={r['bits_per_byte']}")
        except Exception as e:
            r = {k: "" for k in FIELDS}
            r["label"] = label_for(name)
            r["experiment"] = experiment_for(name)
            r["checkpoint_dir"] = name
            r["error"] = f"{type(e).__name__}: {e}"
            log(f"ERR {name:26s} {r['error']}")
        rows.append(r)

    for b in BASELINES:                                  # append external anchors (not loaded)
        r = {k: "" for k in FIELDS}
        r.update(b)
        r["steps"] = ""
        r["converged"] = ""
        rows.append(r)
        log(f"BASE {b['label']:22s} p={b['params_total']:>8d} bpb={b['bits_per_byte']}")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    log(f"\nwrote {len(rows)} rows -> {OUT_CSV}")

    spot_check(rows)


def spot_check(rows):  # reproduce old bpb_all.csv rows + key.json non_embed
    by_label = {r["label"]: r for r in rows if not r["error"]}
    print("\n=== SPOT-CHECK vs data/bpb_all.csv (val_loss / bpb) ===")
    ref = {  # label: (val_loss, bpb) from the original clean rows of bpb_all.csv
        "J": (1.1052, 0.7674), "H": (1.2342, 0.8570), "G": (1.3106, 0.9101),
        "D": (1.3192, 0.9161), "I": (1.3769, 0.9561), "E": (1.4456, 1.0038),
        "A": (1.5686, 1.0892), "F": (1.8700, 1.2985), "B": (2.1156, 1.0022),
        "C": (2.2848, 0.9122),
    }
    print(f"{'label':6s}{'val(new)':>10s}{'val(old)':>10s}{'dv':>8s}"
          f"{'bpb(new)':>10s}{'bpb(old)':>10s}{'db':>8s}")
    for lab, (vl, bp) in ref.items():
        r = by_label.get(lab)
        if not r:
            print(f"{lab:6s}  MISSING")
            continue
        dv, db = r["val_loss"] - vl, r["bits_per_byte"] - bp
        flag = "  <-- CHECK" if (abs(dv) > 0.01 or abs(db) > 0.005) else ""
        print(f"{lab:6s}{r['val_loss']:>10.4f}{vl:>10.4f}{dv:>+8.4f}"
              f"{r['bits_per_byte']:>10.4f}{bp:>10.4f}{db:>+8.4f}{flag}")

    print("\n=== SPOT-CHECK non_embed vs eval/grade_frontier/key.json ===")
    keyp = os.path.join(HERE, "eval", "grade_frontier", "key.json")
    try:
        with open(keyp, encoding="utf-8") as f:
            key = json.load(f)
        seen = {}
        for v in key.values():
            seen[(v["params"], v["non_embed"])] = v["model"]
        by_params = {}
        for r in rows:
            if not r["error"]:
                by_params.setdefault(r["params_total"], r)
        for (p, ne), mdl in sorted(seen.items()):
            r = by_params.get(p)
            got = r["params_non_embed"] if r else None
            flag = "" if got == ne else "  <-- CHECK"
            print(f"{mdl:34s} params={p:>8d} key.non_embed={ne:>8d} ours={got}{flag}")
    except FileNotFoundError:
        print("key.json not found; skipped")


if __name__ == "__main__":
    main()
