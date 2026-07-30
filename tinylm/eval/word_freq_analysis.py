"""Test the hypothesis: are the tokenizer's pieces well-allocated to common whole words?
Analyzes TinyStories word-frequency distribution and how many pieces the current no-fallback
tokenizer spends per common word."""
import json, os, re, collections
import sentencepiece as spm

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
sp = spm.SentencePieceProcessor(); sp.load(os.path.join(DATA, "tok512dep.model"))

stories = json.load(open(os.path.join(DATA, "TinyStories_all_data", "data00.json"), encoding="utf-8"))
text = "\n".join(s["story"] for s in stories[:20000])
words = re.findall(r"[a-zA-Z']+", text.lower())
freq = collections.Counter(words)
total = sum(freq.values())
uniq = len(freq)
print(f"corpus: {total:,} word tokens, {uniq:,} unique words (20k stories)")

# coverage curve: how many distinct words to cover X% of running text
cum = 0; marks = {0.5:None,0.8:None,0.9:None,0.95:None,0.99:None}
for i,(w,c) in enumerate(freq.most_common(),1):
    cum += c
    for th in marks:
        if marks[th] is None and cum/total >= th: marks[th]=i
print("distinct words needed to cover:", {f"{int(k*100)}%":v for k,v in sorted(marks.items())})

# how many tokenizer pieces per common word (with leading space, as in running text)
def pieces_for(word):
    return len(sp.encode("▁"+word)) if False else len(sp.encode(" "+word))
for topn in [50,100,200,300,512,1000,1500]:
    top = [w for w,_ in freq.most_common(topn)]
    single = sum(1 for w in top if pieces_for(w)==1)
    avg = sum(pieces_for(w) for w in top)/len(top)
    print(f"top {topn:>4} words: {single:>4} are a single token ({100*single/len(top):4.1f}%), avg pieces/word {avg:.2f}")

# weighted: average pieces per word occurrence across ALL running text (the real cost)
wpieces = sum(freq[w]*pieces_for(w) for w in freq)/total
print(f"\nweighted avg pieces per word-occurrence (whole corpus): {wpieces:.3f}")
print(f"(1.0 = every word is one token; higher = words being split)")

# which HIGH-FREQUENCY words are being split (wasteful)?
print("\ntop split words (frequent words that cost >1 token):")
split = [(w,freq[w],pieces_for(w)) for w,_ in freq.most_common(400) if pieces_for(w)>1]
split.sort(key=lambda x:-x[1])
for w,c,p in split[:25]:
    print(f"  {w!r:15} freq {c:>6}  -> {p} tokens  {sp.encode(' '+w, out_type=str)}")
print(f"\n{len(split)} of the top-400 words are split into multiple tokens")
