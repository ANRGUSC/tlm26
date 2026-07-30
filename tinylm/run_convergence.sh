#!/bin/sh
# First convergence runs: resume d48/10L to 30k + train QAT int4 dep to 30k.
cd "C:/Users/Mhlit/Desktop/Coding/TinyStories recreation/tinylm" || exit 1
PY=C:/Users/Mhlit/tinystories/.venv/Scripts/python.exe
COMMON="--vocab_source=custom --vocab_size=512 --n_heads=8 --n_kv_heads=4 --batch_size=128 --max_iters=30000 --eval_interval=1000 --compile=False"

echo "[resume 2/3] d48/10L deep-narrow from step 16000 -> 30000"
$PY train.py $COMMON --dim=48 --n_layers=10 --init_from=resume \
    --out_dir=out_iso_d48L10_long >> data/train_iso_d48L10_long.log 2>&1
echo "[2/3 exit=$?] $(grep -E '^step 30000: ' data/train_iso_d48L10_long.log | tail -1)"

echo "[3/3] QAT int4 dep -> convergence"
TS_TOK_NAME=tok512dep $PY train.py $COMMON --dim=64 --n_layers=5 --qat_level=int4 \
    --out_dir=out_qat_int4_dep_long > data/train_qat_int4_dep_long.log 2>&1
echo "[3/3 exit=$?] $(grep -E '^step 30000: ' data/train_qat_int4_dep_long.log | tail -1)"

echo "CONVERGENCE_RUNS_ALL_DONE"
