# RWKV-7 training: patches train.py to build an RWKVTransformer. Consumes --head_size.
import sys

_head_size = 32
kept = []
for a in sys.argv:
    if a.startswith('--head_size='):
        _head_size = int(a.split('=')[1])
    else:
        kept.append(a)
sys.argv = kept

print("=" * 50)
print(f"  RWKV-7 (x070) EXPERIMENT: head_size={_head_size}")
print("=" * 50)

import model as model_module
from model_rwkv import RWKVTransformer, RWKVModelArgs


def _make_rwkv(ma):
    # ma is a model.ModelArgs instance built by train.py from its model_args dict.
    assert ma.dim % _head_size == 0, \
        f"dim ({ma.dim}) must be divisible by head_size ({_head_size})"
    args = RWKVModelArgs(
        dim=ma.dim, n_layers=ma.n_layers,
        n_heads=ma.dim // _head_size,          # keep estimate_mfu coherent (Q = head_size)
        n_kv_heads=ma.n_kv_heads, vocab_size=ma.vocab_size,
        hidden_dim=ma.hidden_dim, multiple_of=ma.multiple_of,
        norm_eps=ma.norm_eps, max_seq_len=ma.max_seq_len, dropout=ma.dropout,
        head_size=_head_size,
    )
    return RWKVTransformer(args)


class _Proxy:
    def __call__(self, ma):
        return _make_rwkv(ma)


model_module.Transformer = _Proxy()

# inject head_size into the saved checkpoint's model_args
_orig_save = __import__('torch').save
def _save(obj, f, *a, **k):
    if isinstance(obj, dict) and 'model_args' in obj:
        obj['model_args']['head_size'] = _head_size
    return _orig_save(obj, f, *a, **k)
import torch
torch.save = _save

# skip the llama2.c .bin export (incompatible with RWKV weights)
import export as export_module
export_module.model_export = lambda *a, **k: print("  (skipping .bin export for RWKV model)")

exec(open("train.py").read())
