"""Scrutinize the no-fallback deployment tokenizer (tok512dep) for coverage, roundtrip
fidelity, unk rate on real story text, and vocab composition. Compares to the byte-fallback
baseline (tok512)."""
import json, os, sys, collections
import sentencepiece as spm

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

def load(name):
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(DATA, name))
    return sp

dep = load("tok512dep.model")
base = load("tok512.model")

print("=== VOCAB COMPOSITION (tok512dep) ===")
n = dep.get_piece_size()
types = collections.Counter()
for i in range(n):
    t = dep.id_to_piece(i)
    if dep.is_control(i): types["control"] += 1
    elif dep.is_unknown(i): types["unknown"] += 1
    elif dep.is_unused(i): types["unused"] += 1
    elif len(t) == 1 and t.startswith("<") is False and dep.is_byte(i): types["byte"] += 1
    else: types["normal"] += 1
print(f"vocab_size={n}  unk_id={dep.unk_id()}  bos={dep.bos_id()} eos={dep.eos_id()} pad={dep.pad_id()}")
print("piece types:", dict(types))
# does newline have a real piece?
nl_ids = dep.encode("\n")
print(f"newline '\\n' encodes to ids {nl_ids} -> pieces {[dep.id_to_piece(i) for i in nl_ids]}")

print("\n=== EDGE-CASE ROUNDTRIP (tok512dep) ===")
tests = [
    "Once upon a time, there was a little girl named Lily.",
    'She said, "Hello!" and smiled.',
    "Tom's dog ran fast — very fast.",
    "They had 3 apples and 12 oranges.",
    "Line one.\nLine two.\n",
    "The naive café résumé costs $5.",
    "It was AWESOME!!! Really?!?",
    "co-operate, well-known, mother-in-law",
]
for s in tests:
    ids = dep.encode(s)
    dec = dep.decode(ids)
    unk = sum(1 for i in ids if i == dep.unk_id())
    ok = "OK" if dec == s else "DIFF"
    print(f"[{ok}] unk={unk}  {repr(s)[:50]}")
    if dec != s:
        print(f"      decoded: {repr(dec)[:70]}")

print("\n=== UNK RATE ON REAL STORY TEXT ===")
stories = json.load(open(os.path.join(DATA, "TinyStories_all_data", "data00.json"), encoding="utf-8"))
sample = [s["story"] for s in stories[:3000]]
text = "\n".join(sample)
for name, sp in [("tok512dep(no-fallback)", dep), ("tok512(byte-fallback)", base)]:
    ids = sp.encode(text)
    nunk = sum(1 for i in ids if i == sp.unk_id())
    nbytes = len(text.encode("utf-8"))
    print(f"{name:24s} tokens={len(ids):>8}  unk={nunk:>5} ({100*nunk/len(ids):.3f}%)  tokens/byte={len(ids)/nbytes:.4f}")

# which characters hit unk under dep?
print("\n=== CHARACTERS THAT MAP TO UNK (tok512dep) ===")
bad = collections.Counter()
for ch in set(text):
    if dep.encode(ch) == [dep.unk_id()] or (len(dep.encode(ch))==1 and dep.encode(ch)[0]==dep.unk_id()):
        bad[ch] += text.count(ch)
if bad:
    for ch, c in bad.most_common(20):
        print(f"  {repr(ch)}: {c} occurrences")
else:
    print("  none — every character in the 3000-story sample is covered")
