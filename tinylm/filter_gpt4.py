"""
Filter TinyStories shards to GPT-4-generated stories only (source == "GPT-4"),
for the data-quality experiment. Writes filtered shards under the same names to
data/TinyStories_gpt4/; the originals in data/TinyStories_all_data/ are untouched.

    python filter_gpt4.py --workers 4
"""
import argparse
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

IN_DIR = os.path.join("data", "TinyStories_all_data")
OUT_DIR = os.path.join("data", "TinyStories_gpt4")


def filter_shard(shard):
    with open(shard, "r", encoding="utf-8") as f:
        data = json.load(f)
    kept = [ex for ex in data if ex.get("source") == "GPT-4"]
    out = os.path.join(OUT_DIR, os.path.basename(shard))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(kept, f, ensure_ascii=False)
    print(f"{os.path.basename(shard)}: kept {len(kept):,}/{len(data):,}", flush=True)
    return len(kept), len(data)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    shards = sorted(glob.glob(os.path.join(IN_DIR, "*.json")))
    assert shards, "no data shards found"
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(filter_shard, shards))
    kept = sum(r[0] for r in results)
    total = sum(r[1] for r in results)
    print(f"TOTAL: kept {kept:,}/{total:,} stories ({kept/total:.1%})")
