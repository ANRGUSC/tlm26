#!/bin/sh
# MoE robustness: top-1/8e (matched-active vs dense d64) + top-2/16e (more sparse capacity), 30k each.
cd "C:/Users/Mhlit/Desktop/Coding/TinyStories recreation/tinylm" || exit 1
PY=C:/Users/Mhlit/tinystories/.venv/Scripts/python.exe
COMMON="--vocab_source=custom --vocab_size=512 --dim=64 --n_layers=5 --n_heads=8 --n_kv_heads=4 --batch_size=128 --max_seq_len=256 --max_iters=30000 --eval_interval=1000 --compile=False"

echo "[top1/8e] start"
$PY train_moe.py $COMMON --n_experts=8 --moe_top_k=1 --out_dir=out_moe_d64_e8_top1 > data/train_moe_e8_top1.log 2>&1
echo "[top1/8e] exit=$? $(grep -E '^step 30000:' data/train_moe_e8_top1.log | tail -1)"

echo "[top2/16e] start"
$PY train_moe.py $COMMON --n_experts=16 --moe_top_k=2 --out_dir=out_moe_d64_e16 > data/train_moe_e16.log 2>&1
echo "[top2/16e] exit=$? $(grep -E '^step 30000:' data/train_moe_e16.log | tail -1)"

echo "MOE_ROBUSTNESS_DONE"
