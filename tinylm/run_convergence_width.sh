#!/bin/sh
# Retrain the width sweep (d32/d48/d64/d96, 5 layers) to 30k -> out_width_d<dim>_long.
cd "C:/Users/Mhlit/Desktop/Coding/TinyStories recreation/tinylm" || exit 1
PY=C:/Users/Mhlit/tinystories/.venv/Scripts/python.exe
COMMON="--vocab_source=custom --vocab_size=512 --n_heads=8 --n_kv_heads=4 --batch_size=128 --max_seq_len=256 --max_iters=30000 --eval_interval=1000 --compile=False --n_layers=5"

for DIM in 32 48 64 96; do
  OUT=out_width_d${DIM}_long
  LOG=data/train_width_d${DIM}_long.log
  echo "[width d${DIM}] start -> $OUT"
  $PY train.py $COMMON --dim=${DIM} --out_dir=$OUT > $LOG 2>&1
  echo "[width d${DIM}] exit=$? $(grep -E '^step 30000: ' $LOG | tail -1)"
done

echo "WIDTH_CONVERGENCE_ALL_DONE"
