"""
LLM-as-judge grading, replicating the TinyStories (Eldan & Li, 2023) GPT-4 rubric
with Claude Sonnet. Each model completion is graded on grammar, creativity,
consistency, and plot (1-10), plus an estimated author age group (A-F).

Auth: set ANTHROPIC_API_KEY (or `ant auth login`). Model: claude-sonnet-5.

    python eval/llm_judge.py                 # grades eval/completions.json
Outputs eval/graded.json (per-completion) and prints per-model means.
"""
import json
import os
from collections import defaultdict

import anthropic
from pydantic import BaseModel, Field

HERE = os.path.dirname(os.path.abspath(__file__))
JUDGE_MODEL = "claude-sonnet-5"

# Faithful to the TinyStories GPT-4 grading method (arXiv:2305.07759), adapted to
# return structured scores. *** separates the given beginning from the completion.
RUBRIC = """In the following exercise, a student is given the beginning of a short \
story written for 3-4 year olds. The student must complete it into a full story. \
The exercise tests the student's language abilities and creativity. The symbol *** \
marks the separator between the prescribed beginning and the student's completion:

{beginning} ***{completion}

Assess ONLY the part the student wrote (after the ***). Pay special attention to \
whether the student correctly completes the sentence that is split by the *** \
separator, whether it is grammatical, and whether it stays consistent with the \
beginning. Then grade the completion on each axis as an integer from 1 to 10:
- grammar: is the writing grammatical and are the words real, correctly-spelled words?
- creativity: is the continuation imaginative and interesting?
- consistency: does it follow logically and consistently from the given beginning?
- plot: does the story make sense and have a coherent plot?
Also give your best guess of the author's age group, reflecting the writing ability:
A: 3 or under, B: 4-5, C: 6-7, D: 8-9, E: 10-12, F: 13-16."""


class Grade(BaseModel):
    grammar: int = Field(ge=1, le=10)
    creativity: int = Field(ge=1, le=10)
    consistency: int = Field(ge=1, le=10)
    plot: int = Field(ge=1, le=10)
    age_group: str = Field(description="one of A,B,C,D,E,F")
    comment: str = Field(description="one short sentence justifying the scores")


def load_env():
    """Load KEY=VALUE lines from eval/.env into the environment (no dependency)."""
    path = os.path.join(HERE, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main():
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY (env var or eval/.env) before running.")
    client = anthropic.Anthropic()  # resolves ANTHROPIC_API_KEY / ant profile
    items = json.load(open(os.path.join(HERE, "completions.json"), encoding="utf-8"))
    graded = []
    for n, it in enumerate(items, 1):
        prompt = RUBRIC.format(beginning=it["prompt"], completion=it["completion"])
        resp = client.messages.parse(
            model=JUDGE_MODEL, max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            output_format=Grade,
        )
        g = resp.parsed_output
        graded.append({**it, **g.model_dump()})
        print(f"[{n}/{len(items)}] {it['model']}: "
              f"gram={g.grammar} crea={g.creativity} cons={g.consistency} plot={g.plot} age={g.age_group}",
              flush=True)

    with open(os.path.join(HERE, "graded.json"), "w", encoding="utf-8") as f:
        json.dump(graded, f, indent=2, ensure_ascii=False)

    # per-model means
    agg = defaultdict(lambda: defaultdict(list))
    for g in graded:
        for k in ("grammar", "creativity", "consistency", "plot"):
            agg[g["model"]][k].append(g[k])
    print("\n=== Per-model means (1-10) ===")
    print("model,grammar,creativity,consistency,plot,overall")
    for model, d in agg.items():
        means = {k: sum(v) / len(v) for k, v in d.items()}
        overall = sum(means.values()) / 4
        print(f"{model},{means['grammar']:.2f},{means['creativity']:.2f},"
              f"{means['consistency']:.2f},{means['plot']:.2f},{overall:.2f}")


if __name__ == "__main__":
    main()
