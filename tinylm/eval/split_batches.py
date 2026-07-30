"""
Split completions_data_faithful.json into fixed-size batch files for parallel
subagent grading. Writes eval/grade_batches/batch_00.json ... and prints the count.

    python eval/split_batches.py [batch_size]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 100

src = json.load(open(os.path.join(HERE, "completions_data_faithful.json"), encoding="utf-8"))
outdir = os.path.join(HERE, "grade_batches")
os.makedirs(outdir, exist_ok=True)
# clear old batches
for f in os.listdir(outdir):
    if f.startswith("batch_") or f.startswith("scores_"):
        os.remove(os.path.join(outdir, f))

# attach a global id so scores can be mapped back
for i, x in enumerate(src):
    x["id"] = i

n = 0
for b in range(0, len(src), BATCH_SIZE):
    batch = [{"id": x["id"], "model": x["model"], "prompt": x["prompt"], "completion": x["completion"]}
             for x in src[b:b + BATCH_SIZE]]
    with open(os.path.join(outdir, f"batch_{n:02d}.json"), "w", encoding="utf-8") as f:
        json.dump(batch, f, ensure_ascii=False, indent=1)
    n += 1

print(f"total_completions={len(src)} num_batches={n} batch_size={BATCH_SIZE}")
