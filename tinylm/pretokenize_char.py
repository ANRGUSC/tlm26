"""
Torch-free pretokenizer. tinystories.py imports torch at module top, so on
Windows its ProcessPoolExecutor workers each re-import torch and can crash with
"paging file too small". This script imports ONLY sentencepiece (via Tokenizer),
so workers stay light. Use for any custom tok{N}.

    python pretokenize_char.py --vocab_size 105 --workers 4
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np

from tokenizer import Tokenizer

DATA = "data"


def process_shard(args, tok_model, out_dir):
    shard_id, shard = args
    enc = Tokenizer(tok_model)
    with open(shard, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_tokens = []
    for ex in data:
        text = ex["story"].strip()
        all_tokens.extend(enc.encode(text, bos=True, eos=False))
    arr = np.array(all_tokens, dtype=np.uint16)
    base = os.path.basename(shard).replace(".json", ".bin")
    out = os.path.join(out_dir, base)
    with open(out, "wb") as f:
        f.write(arr.tobytes())
    avg = arr.size / max(1, int((arr == 1).sum()))
    print(f"{base}: {arr.size:,} tokens, avg story len {avg:.1f} tok", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab_size", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--input_dir", default=os.path.join(DATA, "TinyStories_all_data"))
    ap.add_argument("--tokenizer", default=None, help="tokenizer .model path; default data/<tok_name>.model")
    a = ap.parse_args()

    # TS_TOK_NAME disambiguates tokenizers sharing a vocab size (see tinystories.tok_name)
    tok_name = os.environ.get("TS_TOK_NAME", f"tok{a.vocab_size}")
    tok_model = a.tokenizer or os.path.join(DATA, f"{tok_name}.model")
    out_dir = os.path.join(DATA, tok_name)
    os.makedirs(out_dir, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(a.input_dir, "*.json")))
    assert shards, "no data shards found"
    print(f"Pretokenizing {len(shards)} shards with {tok_model} using {a.workers} workers")
    fun = partial(process_shard, tok_model=tok_model, out_dir=out_dir)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        list(ex.map(fun, enumerate(shards)))
    print("Done.")
