"""Per-story ("native") bits-per-byte: each story is scored independently with the
model's own start-of-story token and a fresh context, then loss is normalized by the
story's UTF-8 byte length. This is the fair way to compare models that use different
story-boundary conventions (e.g. TinyStories-656K depends on a <|start_story|> token);
a single continuous token stream penalizes such models unfairly.

Scores HF causal LMs and this project's own checkpoints through one identical path.
"""
import argparse, json, math, os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--hf", type=str, default=None)
ap.add_argument("--out_dir", type=str, default=None)
ap.add_argument("--tok", type=str, default="data/tok512.model")
ap.add_argument("--n_stories", type=int, default=2000)
ap.add_argument("--tag", type=str, default=None)
args = ap.parse_args()

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_stories(path, n):
    if path.endswith(".json"):
        return [x["story"].strip() for x in json.load(open(path, encoding="utf-8"))[:n]]
    return [t.strip() for t in open(path, encoding="utf-8").read().split("<|endoftext|>") if t.strip()][:n]


EVAL = [
    ("data00", os.path.join(ROOT, "data", "TinyStories_all_data", "data00.json")),
    ("V1-valid", os.path.join(ROOT, "data", "TinyStories-valid.txt")),
    ("V2-valid", os.path.join(ROOT, "data", "TinyStoriesV2-GPT4-valid.txt")),
]


def score_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.hf)
    m = AutoModelForCausalLM.from_pretrained(args.hf).to(DEV).eval()
    n = sum(p.numel() for p in m.parameters())
    # story-start context token: use the tokenizer's own start if it adds one, else eos
    adds_bos = tok.encode("hi")[0] == (tok.bos_token_id if tok.bos_token_id is not None else -999)
    start = None if adds_bos else (tok.eos_token_id if tok.eos_token_id is not None else None)
    def enc(s):
        ids = tok.encode(s)
        if start is not None:
            ids = [start] + ids
        return ids
    return m, enc, n


def score_ours():
    from tokenizer import Tokenizer
    from model import ModelArgs, Transformer
    enc_t = Tokenizer(os.path.join(ROOT, args.tok))
    ck = torch.load(os.path.join(ROOT, args.out_dir, "ckpt.pt"), map_location=DEV)
    m = Transformer(ModelArgs(**ck["model_args"]))
    sd = ck["model"]
    for k in list(sd.keys()):
        if k.startswith("_orig_mod."):
            sd[k[len("_orig_mod."):]] = sd.pop(k)
    m.load_state_dict(sd); m.to(DEV).eval()
    n = sum(p.numel() for p in m.parameters())
    def enc(s):
        return enc_t.encode(s, bos=True, eos=False)  # BOS=1 is the story-start context
    return m, enc, n


def main():
    is_hf = args.hf is not None
    m, enc, nparams = score_hf() if is_hf else score_ours()
    label = args.tag or (args.hf or args.out_dir)
    print(f"model: {label}  params={nparams:,}")
    for name, path in EVAL:
        if not os.path.exists(path):
            print(f"  {name}: (missing)"); continue
        stories = load_stories(path, args.n_stories)
        tot_nll, tot_bytes, tot_tok = 0.0, 0, 0
        CAP = 256  # equal context budget for every model (our models' trained seq_len)
        with torch.no_grad():
            for s in stories:
                ids = enc(s)
                if len(ids) < 2:
                    continue
                # tile the story's tokens into <=CAP-token windows (input length <= CAP so
                # the 256-context models fit), with next-token targets aligned so every token
                # after the first is predicted exactly once, no cross-story context
                for o in range(0, len(ids) - 1, CAP):
                    xi = ids[o:o + CAP]                 # <= CAP input tokens
                    yi = ids[o + 1:o + 1 + len(xi)]     # the next token for each input position
                    xi = xi[:len(yi)]
                    if len(xi) == 0:
                        continue
                    x = torch.tensor([xi], dtype=torch.long, device=DEV)
                    y = torch.tensor([yi], dtype=torch.long, device=DEV)
                    if is_hf:
                        logits = m(x).logits[0]
                    else:
                        logits = m(x, x)[0]            # our forward returns full logits when targets given
                    nll = torch.nn.functional.cross_entropy(logits.float(), y[0], reduction="sum")
                    tot_nll += nll.item(); tot_tok += y.numel()
                tot_bytes += len(s.encode("utf-8"))
        print(f"  {name:9s}: bpb={tot_nll/tot_bytes/math.log(2):.4f}  tok/byte={tot_tok/tot_bytes:.4f}")


if __name__ == "__main__":
    main()
