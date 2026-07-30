"""
Rejoin the blinded panel scores to model identities and report per-model means.

Reports each judge separately as well as the panel mean, because a single panel mean can
hide one judge disagreeing with the other two — and this project has already been bitten
once by trusting a judge aggregate without checking dispersion (verdicts_tokcompare.md).

    python eval/aggregate_baseline.py
"""
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
GDIR = os.path.join(HERE, "grade_baseline")
AXES = ["grammar", "creativity", "consistency", "plot"]

key = json.load(open(os.path.join(GDIR, "key.json"), encoding="utf-8"))
judges = {}
for j in (1, 2, 3):
    p = os.path.join(GDIR, f"scores_judge{j}.json")
    if os.path.exists(p):
        judges[j] = {str(r["id"]): r for r in json.load(open(p, encoding="utf-8"))}
    else:
        print(f"WARNING: scores_judge{j}.json missing")

# per-model, per-judge, per-axis
acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for jid, scores in judges.items():
    for iid, rec in scores.items():
        model = key[iid]["model"]
        for ax in AXES:
            acc[model][jid][ax].append(rec[ax])

order = sorted(acc, key=lambda m: next(v["params"] for v in key.values() if v["model"] == m), reverse=True)

print(f"judges: {sorted(judges)}   items/judge: {[len(v) for v in judges.values()]}\n")
hdr = f"{'model':<38}{'params':>9}  " + "".join(f"{a[:5]:>7}" for a in AXES) + f"{'OVERALL':>9}{'spread':>8}"
print(hdr)
print("-" * len(hdr))

rows = []
for m in order:
    params = next(v["params"] for v in key.values() if v["model"] == m)
    axis_means = {}
    for ax in AXES:
        per_judge = [statistics.mean(acc[m][j][ax]) for j in judges]
        axis_means[ax] = statistics.mean(per_judge)
    per_judge_overall = [
        statistics.mean([statistics.mean(acc[m][j][ax]) for ax in AXES]) for j in judges
    ]
    overall = statistics.mean(per_judge_overall)
    spread = max(per_judge_overall) - min(per_judge_overall)
    print(f"{m:<38}{params:>9,}  " + "".join(f"{axis_means[a]:>7.2f}" for a in AXES)
          + f"{overall:>9.2f}{spread:>8.2f}")
    rows.append({"model": m, "params": params, **{a: round(axis_means[a], 3) for a in AXES},
                 "overall": round(overall, 3), "judge_spread": round(spread, 3),
                 "per_judge_overall": [round(x, 3) for x in per_judge_overall]})

print("\nper-judge overall (checks for a judge pulling the panel):")
for m in order:
    pj = [statistics.mean([statistics.mean(acc[m][j][ax]) for ax in AXES]) for j in judges]
    print(f"  {m:<38}" + "  ".join(f"J{j}={v:.2f}" for j, v in zip(judges, pj)))

json.dump(rows, open(os.path.join(GDIR, "panel_summary.json"), "w", encoding="utf-8"), indent=1)
print(f"\n-> {os.path.join(GDIR, 'panel_summary.json')}")
