#!/bin/sh
# Scale experts at FIXED active compute (d64, top-2): 32e then 64e, 30k each. Flash grows, RAM/compute fixed.
cd "C:/Users/Mhlit/Desktop/Coding/TinyStories recreation/tinylm" || exit 1
PY=C:/Users/Mhlit/tinystories/.venv/Scripts/python.exe
COMMON="--vocab_source=custom --vocab_size=512 --dim=64 --n_layers=5 --n_heads=8 --n_kv_heads=4 --batch_size=128 --max_seq_len=256 --max_iters=30000 --eval_interval=1000 --compile=False --moe_top_k=2"

echo "[e32] start"
$PY train_moe.py $COMMON --n_experts=32 --out_dir=out_moe_d64_e32 > data/train_moe_e32.log 2>&1
echo "[e32] exit=$? $(grep -E '^step 30000:' data/train_moe_e32.log | tail -1)"

echo "[e64] start"
$PY train_moe.py $COMMON --n_experts=64 --out_dir=out_moe_d64_e64 > data/train_moe_e64.log 2>&1
echo "[e64] exit=$? $(grep -E '^step 30000:' data/train_moe_e64.log | tail -1)"

echo "MOE_SCALE_DONE"
