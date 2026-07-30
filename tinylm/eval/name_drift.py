"""Quantitative name-drift metric for the over-encoding experiment. Tests the hypothesis that
whole-word input embeddings help the model keep character names stable. Builds a name gazetteer
from the corpus (frequent mid-sentence capitalized words), then for each completion counts:
  - retention: fraction of names introduced in the PROMPT that reappear in the completion
  - foreign:  count of gazetteer names in the completion that were NOT in the prompt (drift/intrusion)
Lower foreign + higher retention = more stable characters."""
import json, os, re, collections
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# build a name gazetteer: words that appear capitalized MID-sentence (proper nouns), frequency-ranked
stories = json.load(open(os.path.join(DATA, "TinyStories_all_data", "data00.json"), encoding="utf-8"))
text = " ".join(s["story"] for s in stories[:20000])
# tokens not at sentence start: split into sentences, drop first word of each
caps = collections.Counter()
for sent in re.split(r'(?<=[.!?])\s+', text):
    toks = re.findall(r"[A-Za-z]+", sent)
    for w in toks[1:]:                      # skip sentence-initial capital
        if w[0].isupper() and w[1:].islower() and len(w) >= 3:
            caps[w] += 1
STOP = {"The","And","But","She","Her","His","They","Then","One","Mom","Dad","Mommy","Daddy","Mr","Mrs","Miss","When","After","Once","There","Suddenly","Now","Yes","With","This","That",
        "Can","From","Hello","Let","Look","Okay","Please","Sure","Thank","What","Why","Wow","You","Don","Him","Her","Its","But","So","Oh","Well","Here","Have","Get","Come","Good","Are","Was","Not"}
gaz = {w for w,_ in caps.most_common(80) if w not in STOP}
print(f"gazetteer ({len(gaz)}): {sorted(gaz)}\n")

def names_in(s):
    return {w for w in re.findall(r"[A-Za-z]+", s) if w in gaz}

data = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "completions_oe.json"), encoding="utf-8"))
by_model = collections.defaultdict(list)
for e in data: by_model[e["model"]].append(e)

print(f"{'model':32s} {'ret%':>6} {'foreign/compl':>14} {'intruded%':>10}")
for model, es in by_model.items():
    rets, foreigns, intruded = [], [], 0
    for e in es:
        intro = names_in(e["prompt"]); comp = names_in(e["completion"])
        if intro:
            rets.append(len(intro & comp) / len(intro))
        foreign = comp - intro
        foreigns.append(len(foreign))
        if foreign: intruded += 1
    ret = 100*sum(rets)/len(rets) if rets else float('nan')
    print(f"{model:32s} {ret:6.1f} {sum(foreigns)/len(es):14.2f} {100*intruded/len(es):10.1f}")
