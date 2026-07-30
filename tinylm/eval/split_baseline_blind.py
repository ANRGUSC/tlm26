"""
Split completions_baseline.json into a BLINDED, shuffled grading file for the merged panel.

`split_batches.py` passes the `model` field through to the judge. That is tolerable when the
candidates are all this project's own checkpoints, but this panel compares against published
baselines, and a judge that can read "TinyStories-1M (paper baseline)" next to "ours" is being
handed the answer. Here the model label is stripped and the items are shuffled under a fixed
seed, so a judge sees only prompt+completion. The key mapping id -> model stays local and is
rejoined after grading.

    python eval/split_baseline_blind.py
      -> eval/grade_baseline/blind_items.json   (given to judges: id, prompt, completion)
      -> eval/grade_baseline/key.json           (NOT given to judges: id -> model)
"""
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 1337

src = json.load(open(os.path.join(HERE, "completions_baseline.json"), encoding="utf-8"))
outdir = os.path.join(HERE, "grade_baseline")
os.makedirs(outdir, exist_ok=True)

items = []
key = {}
for i, x in enumerate(src):
    items.append({"id": i, "prompt": x["prompt"], "completion": x["completion"]})
    key[str(i)] = {"model": x["model"], "params": x["params"], "non_embed": x["non_embed"]}

random.Random(SEED).shuffle(items)  # so judges cannot infer identity from ordering

json.dump(items, open(os.path.join(outdir, "blind_items.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
json.dump(key, open(os.path.join(outdir, "key.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)

print(f"blinded_items={len(items)} models={len(set(v['model'] for v in key.values()))}")
print(f"-> {os.path.join(outdir, 'blind_items.json')}")
