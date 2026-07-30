"""
Looped Transformer training script.
Patches train.py to use LoopedTransformer instead of Transformer.

Usage:
  python train_looped.py --n_unique_layers=2 --n_loops=3 --dim=96 --vocab_source=custom --vocab_size=512 ...

The --n_unique_layers and --n_loops flags are consumed here; all other flags
are passed through to train.py's configurator.
"""
import sys
import builtins

# --- Extract looped-transformer args before configurator sees them ---
_n_unique_layers = 2
_n_loops = 3
_input_injection = False

filtered_argv = []
for arg in sys.argv:
    if arg.startswith('--n_unique_layers='):
        _n_unique_layers = int(arg.split('=')[1])
    elif arg.startswith('--n_loops='):
        _n_loops = int(arg.split('=')[1])
    elif arg.startswith('--input_injection='):
        _input_injection = arg.split('=')[1].lower() in ('1', 'true', 'yes')
    else:
        filtered_argv.append(arg)
sys.argv = filtered_argv

effective_depth = _n_unique_layers * _n_loops
residual_scale = 1.0 / _n_loops

print("=" * 50)
print(f"  LOOPED TRANSFORMER EXPERIMENT")
print(f"  Unique layers: {_n_unique_layers}")
print(f"  Loops: {_n_loops}")
print(f"  Effective depth: {effective_depth}")
print(f"  Residual scale: 1/{_n_loops} = {residual_scale:.4f}")
print("=" * 50)

# --- Patch the model module ---
import model as model_module
from model_looped import LoopedTransformer, LoopedModelArgs

_OrigModelArgs = model_module.ModelArgs

def _make_looped_transformer(model_args_instance):
    looped_args = LoopedModelArgs(
        dim=model_args_instance.dim,
        n_layers=model_args_instance.n_layers,
        n_heads=model_args_instance.n_heads,
        n_kv_heads=model_args_instance.n_kv_heads,
        vocab_size=model_args_instance.vocab_size,
        hidden_dim=model_args_instance.hidden_dim,
        multiple_of=model_args_instance.multiple_of,
        norm_eps=model_args_instance.norm_eps,
        max_seq_len=model_args_instance.max_seq_len,
        dropout=model_args_instance.dropout,
        n_unique_layers=_n_unique_layers,
        n_loops=_n_loops,
        input_injection=_input_injection,
    )
    m = LoopedTransformer(looped_args)
    total_params = sum(p.numel() for p in m.parameters())
    print(f"  Looped model params: {total_params:,}")
    return m

class _TransformerProxy:
    def __call__(self, model_args):
        return _make_looped_transformer(model_args)

model_module.Transformer = _TransformerProxy()

# Patch torch.save so checkpoint includes our looped params
_orig_torch_save = __import__('torch').save

def _patched_torch_save(obj, f, *args, **kwargs):
    """Inject n_unique_layers and n_loops into checkpoint's model_args dict."""
    if isinstance(obj, dict) and 'model_args' in obj:
        obj['model_args']['n_unique_layers'] = _n_unique_layers
        obj['model_args']['n_loops'] = _n_loops
        obj['model_args']['input_injection'] = _input_injection
    return _orig_torch_save(obj, f, *args, **kwargs)

import torch
torch.save = _patched_torch_save

# Patch model_export to skip .bin export (format doesn't support looped models)
import export as export_module
_orig_export = export_module.model_export
def _skip_export(*args, **kwargs):
    print("  (skipping .bin export for looped model)")
export_module.model_export = _skip_export

# --- Run the original train.py ---
exec(open("train.py").read())
