# Generate from a trained RWKV-7 checkpoint (coherence check).
import dataclasses
import sys

import torch

from model_rwkv import RWKVModelArgs, RWKVTransformer
from tokenizer import Tokenizer
from tinystories import get_tokenizer_model_path

ckpt = sys.argv[1] if len(sys.argv) > 1 else "out_rwkv_d64_long/ckpt.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"

cd = torch.load(ckpt, map_location=device, weights_only=False)
ma = cd["model_args"]
fields = {f.name for f in dataclasses.fields(RWKVModelArgs)}
model = RWKVTransformer(RWKVModelArgs(**{k: v for k, v in ma.items() if k in fields}))
sd = cd["model"]
for k in list(sd.keys()):
    if k.startswith("_orig_mod."):
        sd[k[len("_orig_mod."):]] = sd.pop(k)
model.load_state_dict(sd, strict=False)
model.eval().to(device)

enc = Tokenizer(get_tokenizer_model_path(vocab_size=int(ma["vocab_size"])))
prompts = ["Once upon a time, there was a little robot",
           "The cat and the dog wanted to play",
           "One day, a girl named Lily found a magic key"]
with torch.no_grad():
    for p in prompts:
        ids = enc.encode(p, bos=True, eos=False)
        x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
        y = model.generate(x, 90, temperature=0.8, top_k=200)
        print(">>", enc.decode(y[0].tolist()))
        print("-" * 60)
