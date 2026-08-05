#!/bin/sh
# RWKV-7 vs transformer comparison at a matched reduced budget (dim64/5L, seq256, batch64, 20k).
# The transformer control removes the confound of RWKV's smaller-than-headline budget.
# bpb = val_loss * tokens_per_byte(0.4813 for tok512) / ln2.
cd "C:/Users/Mhlit/Desktop/Coding/TinyStories recreation/tinylm" || exit 1
PY=C:/Users/Mhlit/tinystories/.venv/Scripts/python.exe
COMMON="--vocab_source=custom --vocab_size=512 --dim=64 --n_layers=5 --n_heads=8 --n_kv_heads=4 --batch_size=64 --max_seq_len=256 --max_iters=20000 --eval_interval=1000 --compile=False"
LOG=data/rwkv_comparison.log
: > $LOG
echo "=== transformer control (matched budget) ===" | tee -a $LOG
$PY train.py $COMMON --out_dir=out_ctrl_d64_b64_long > data/train_ctrl_d64_b64.log 2>&1
echo "[ctrl] exit=$? $(grep -E '^step 20000:' data/train_ctrl_d64_b64.log | tail -1)" | tee -a $LOG

echo "=== RWKV-7 (same budget) ===" | tee -a $LOG
$PY train_rwkv.py $COMMON --head_size=32 --out_dir=out_rwkv_d64_long > data/train_rwkv_d64.log 2>&1
echo "[rwkv] exit=$? $(grep -E '^step 20000:' data/train_rwkv_d64.log | tail -1)" | tee -a $LOG

$PY - <<'EOF' | tee -a $LOG
import re
LN2 = 0.6931471805599453; TPB = 0.4813
print("=== comparison bpb ===")
for name, path in [("transformer ctrl", "data/train_ctrl_d64_b64.log"), ("RWKV-7", "data/train_rwkv_d64.log")]:
    try:
        m = re.findall(r'step \d+: train loss [\d.]+, val loss ([\d.]+)', open(path).read())
        if m:
            vl = float(m[-1]); print(f"{name:18s} val_loss={vl:.4f}  bpb={vl*TPB/LN2:.4f}")
    except Exception as e:
        print(name, "err", e)
EOF
echo "RWKV_COMPARISON_DONE" | tee -a $LOG
