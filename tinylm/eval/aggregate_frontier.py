"""
Rejoin the blinded frontier panel scores to model identities and report the re-graded frontier.

Prints models in ascending parameter order so the ladder reads bottom-up: the question this panel
exists to answer is whether a coherence ONSET is visible at all once grading is blinded, or whether
quality just rises smoothly with capacity.

Reports each judge separately alongside the panel mean, because a panel mean can conceal one judge
carrying the result (see the calibration-drift finding in verdicts_tokcompare.md).

    python eval/aggregate_frontier.py
"""
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GDIR = os.path.join(HERE, "grade_frontier")
AXES = ["grammar", "creativity", "consistency", "plot"]

key = json.load(open(os.path.join(GDIR, "key.json"), encoding="utf-8"))

# scores_batch{k}_judge{j}.json -> {id: record}
graded = defaultdict(dict)   # judge -> id -> rec
for fn in sorted(os.listdir(GDIR)):
    if not fn.startswith("scores_"):
        continue
    j = fn.split("judge")[1].split(".")[0]
    for r in json.load(open(os.path.join(GDIR, fn), encoding="utf-8")):
        graded[j][str(r["id"])] = r

missing = [i for i in key if not all(i in graded[j] for j in graded)]
print(f"judges: {sorted(graded)}   graded/judge: {[len(v) for v in graded.values()]}"
      f"   items: {len(key)}   items missing a grade: {len(missing)}\n")

acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for j, scores in graded.items():
    for iid, rec in scores.items():
        for ax in AXES:
            acc[key[iid]["model"]][j][ax].append(rec[ax])

order = sorted(acc, key=lambda m: next(v["params"] for v in key.values() if v["model"] == m))

hdr = (f"{'model':<32}{'params':>10}{'non-emb':>9}  " + "".join(f"{a[:5]:>7}" for a in AXES)
       + f"{'OVERALL':>9}{'spread':>8}")
print(hdr)
print("-" * len(hdr))

rows = []
for m in order:
    meta = next(v for v in key.values() if v["model"] == m)
    judges = sorted(acc[m])
    axis_means = {ax: statistics.mean([statistics.mean(acc[m][j][ax]) for j in judges]) for ax in AXES}
    per_judge = [statistics.mean([statistics.mean(acc[m][j][ax]) for ax in AXES]) for j in judges]
    overall = statistics.mean(per_judge)
    spread = max(per_judge) - min(per_judge)
    print(f"{m:<32}{meta['params']:>10,}{meta['non_embed']:>9,}  "
          + "".join(f"{axis_means[a]:>7.2f}" for a in AXES) + f"{overall:>9.2f}{spread:>8.2f}")
    rows.append({"model": m, "params": meta["params"], "non_embed": meta["non_embed"],
                 **{a: round(axis_means[a], 3) for a in AXES},
                 "overall": round(overall, 3), "judge_spread": round(spread, 3)})

print("\nstep-up in OVERALL between adjacent rungs (a coherence ONSET would show as a jump):")
for a, b in zip(rows, rows[1:]):
    print(f"  {a['model']:<32} -> {b['model']:<32} {b['overall']-a['overall']:+.2f}")

json.dump(rows, open(os.path.join(GDIR, "panel_summary.json"), "w", encoding="utf-8"), indent=1)
print(f"\n-> {os.path.join(GDIR, 'panel_summary.json')}")
