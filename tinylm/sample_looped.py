"""
Sample from a trained Looped Transformer model.

Usage:
  python sample_looped.py --checkpoint=out_loop_2x3/ckpt.pt --num_samples=5 --temperature=0.8
"""
import os
import torch
from contextlib import nullcontext
from model_looped import LoopedTransformer, LoopedModelArgs
from tinystories import get_tokenizer_model_path
from tokenizer import Tokenizer

# defaults
checkpoint = 'out/ckpt.pt'
num_samples = 1
max_new_tokens = 100
temperature = 1.0
top_k = 300
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = "float32"
exec(open('configurator.py').read())

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# load checkpoint
checkpoint_dict = torch.load(checkpoint, map_location=device)
model_args = checkpoint_dict['model_args']

# detect looped model args
if 'n_unique_layers' in model_args:
    conf = LoopedModelArgs(**model_args)
    model = LoopedTransformer(conf)
    print(f"Looped model: {conf.n_unique_layers} unique layers x {conf.n_loops} loops = {conf.n_unique_layers * conf.n_loops} effective depth")
else:
    from model import ModelArgs, Transformer
    conf = ModelArgs(**model_args)
    model = Transformer(conf)

state_dict = checkpoint_dict['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict, strict=False)
model.eval()
model.to(device)

# load tokenizer
vocab_source = checkpoint_dict["config"].get("vocab_source", "llama2")
vocab_size = conf.vocab_size
query_vocab_size = 0 if vocab_source == "llama2" else vocab_size
tokenizer_model = get_tokenizer_model_path(vocab_size=query_vocab_size)
enc = Tokenizer(tokenizer_model=tokenizer_model)

# generate
start_ids = enc.encode("", bos=True, eos=False)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(enc.decode(y[0].tolist()))
            print('---------------')
