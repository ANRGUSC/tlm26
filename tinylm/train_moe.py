# MoE training: patches train.py to build a MoETransformer. Consumes --n_experts/--moe_top_k/--aux_coef.
import sys

_n_experts, _top_k, _aux_coef = 8, 2, 0.01
kept = []
for a in sys.argv:
    if a.startswith('--n_experts='):
        _n_experts = int(a.split('=')[1])
    elif a.startswith('--moe_top_k='):
        _top_k = int(a.split('=')[1])
    elif a.startswith('--aux_coef='):
        _aux_coef = float(a.split('=')[1])
    else:
        kept.append(a)
sys.argv = kept

print("=" * 50)
print(f"  MoE EXPERIMENT: {_n_experts} experts, top-{_top_k}, aux_coef={_aux_coef}")
print("=" * 50)

import model as model_module
from model_moe import MoETransformer, MoEModelArgs

def _make_moe(ma):
    args = MoEModelArgs(dim=ma.dim, n_layers=ma.n_layers, n_heads=ma.n_heads, n_kv_heads=ma.n_kv_heads,
                        vocab_size=ma.vocab_size, hidden_dim=ma.hidden_dim, multiple_of=ma.multiple_of,
                        norm_eps=ma.norm_eps, max_seq_len=ma.max_seq_len, dropout=ma.dropout,
                        n_experts=_n_experts, moe_top_k=_top_k, aux_coef=_aux_coef)
    m = MoETransformer(args)
    active = sum(p.numel() for n, p in m.named_parameters()
                 if 'experts.' not in n or any(f'experts.{e}.' in n for e in range(_top_k)))
    total = sum(p.numel() for p in m.parameters())
    print(f"  MoE params: {total:,} total  (~active/token accounting printed at eval)")
    return m

class _Proxy:
    def __call__(self, ma):
        return _make_moe(ma)

model_module.Transformer = _Proxy()

_orig_save = __import__('torch').save
def _save(obj, f, *a, **k):
    if isinstance(obj, dict) and 'model_args' in obj:
        obj['model_args']['n_experts'] = _n_experts
        obj['model_args']['moe_top_k'] = _top_k
        obj['model_args']['aux_coef'] = _aux_coef
    return _orig_save(obj, f, *a, **k)
import torch
torch.save = _save

import export as export_module
export_module.model_export = lambda *a, **k: print("  (skipping .bin export for MoE model)")

exec(open("train.py").read())
