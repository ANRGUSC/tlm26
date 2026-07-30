"""
Blind, shuffle and batch completions_frontier_blind.json for the frontier re-grade panel.

Blinding: the judge receives only id/prompt/completion. `eval/split_batches.py`, used by every
prior panel, passed the model label through — untenable here, where the panel contains published
baselines next to this project's checkpoints.

Batching: 260 items is too many for one judge to grade attentively in a single pass, so the set is
split into batches and each batch is graded by three independent judges. The shuffle (fixed seed)
happens BEFORE batching, so every batch holds a mix of all models. That matters: if a batch's judges
drift in calibration, the drift lands on all models roughly equally and adds noise rather than bias.

    python eval/split_frontier_blind.py [n_batches]
      -> eval/grade_frontier/blind_batch_{k}.json   (judges see these)
      -> eval/grade_frontier/key.json               (judges never see this)
"""
import json
import math
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 1337
N_BATCHES = int(sys.argv[1]) if len(sys.argv) > 1 else 3

src = json.load(open(os.path.join(HERE, "completions_frontier_blind.json"), encoding="utf-8"))
outdir = os.path.join(HERE, "grade_frontier")
os.makedirs(outdir, exist_ok=True)
for f in os.listdir(outdir):
    if f.startswith(("blind_batch_", "scores_")):
        os.remove(os.path.join(outdir, f))

items, key = [], {}
for i, x in enumerate(src):
    items.append({"id": i, "prompt": x["prompt"], "completion": x["completion"]})
    key[str(i)] = {"model": x["model"], "params": x["params"], "non_embed": x["non_embed"]}

random.Random(SEED).shuffle(items)

size = math.ceil(len(items) / N_BATCHES)
for k in range(N_BATCHES):
    batch = items[k * size:(k + 1) * size]
    json.dump(batch, open(os.path.join(outdir, f"blind_batch_{k}.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"blind_batch_{k}.json: {len(batch)} items")

json.dump(key, open(os.path.join(outdir, "key.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"total={len(items)} models={len(set(v['model'] for v in key.values()))} -> {outdir}")
