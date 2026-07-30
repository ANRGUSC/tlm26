#!/bin/sh
# Finish Priority B: resume ternary-d96 from step 10k + train the 4 data-token models.
cd "C:/Users/Mhlit/Desktop/Coding/TinyStories recreation/tinylm" || exit 1
PY=C:/Users/Mhlit/tinystories/.venv/Scripts/python.exe
COMMON="--vocab_source=custom --n_heads=8 --n_kv_heads=4 --batch_size=128 --max_seq_len=256 --max_iters=30000 --eval_interval=1000 --compile=False --n_layers=5"

fin () { echo "[$1] exit=$2 $(grep -E '^step 30000: ' data/$3.log | tail -1)"; }

echo "[ternary_d96 resume] start (from step 10000)"
$PY train.py $COMMON --dim=96 --vocab_size=512 --qat_level=ternary --init_from=resume --out_dir=out_qat_ternary_d96_long >> data/out_qat_ternary_d96_long.log 2>&1
fin ternary_d96 $? out_qat_ternary_d96_long

echo "[t300k] start"; TS_MAX_TRAIN_TOKENS=300000   $PY train.py $COMMON --dim=64 --vocab_size=512 --out_dir=out_data_t300k_long > data/out_data_t300k_long.log 2>&1; fin t300k $? out_data_t300k_long
echo "[t1m] start";   TS_MAX_TRAIN_TOKENS=1000000  $PY train.py $COMMON --dim=64 --vocab_size=512 --out_dir=out_data_t1m_long   > data/out_data_t1m_long.log 2>&1;   fin t1m $? out_data_t1m_long
echo "[t3m] start";   TS_MAX_TRAIN_TOKENS=3000000  $PY train.py $COMMON --dim=64 --vocab_size=512 --out_dir=out_data_t3m_long   > data/out_data_t3m_long.log 2>&1;   fin t3m $? out_data_t3m_long
echo "[t10m] start";  TS_MAX_TRAIN_TOKENS=10000000 $PY train.py $COMMON --dim=64 --vocab_size=512 --out_dir=out_data_t10m_long > data/out_data_t10m_long.log 2>&1; fin t10m $? out_data_t10m_long

echo "PRIORITY_B_RESUME_ALL_DONE"
