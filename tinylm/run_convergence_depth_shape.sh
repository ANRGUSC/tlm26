#!/bin/sh
# Retrain depth (d64, L1-L8) and shape (d80/3L) to 30k -> out_depth_L<k>_long, out_iso_d80L3_long.
cd "C:/Users/Mhlit/Desktop/Coding/TinyStories recreation/tinylm" || exit 1
PY=C:/Users/Mhlit/tinystories/.venv/Scripts/python.exe
COMMON="--vocab_source=custom --vocab_size=512 --n_heads=8 --n_kv_heads=4 --batch_size=128 --max_seq_len=256 --max_iters=30000 --eval_interval=1000 --compile=False"

for L in 1 2 3 4 6 8; do
  OUT=out_depth_L${L}_long
  LOG=data/train_depth_L${L}_long.log
  echo "[depth L${L}] start -> $OUT"
  $PY train.py $COMMON --dim=64 --n_layers=${L} --out_dir=$OUT > $LOG 2>&1
  echo "[depth L${L}] exit=$? $(grep -E '^step 30000: ' $LOG | tail -1)"
done

echo "[shape d80/3L] start -> out_iso_d80L3_long"
$PY train.py $COMMON --dim=80 --n_layers=3 --out_dir=out_iso_d80L3_long > data/train_iso_d80L3_long.log 2>&1
echo "[shape d80/3L] exit=$? $(grep -E '^step 30000: ' data/train_iso_d80L3_long.log | tail -1)"

echo "DEPTH_SHAPE_CONVERGENCE_ALL_DONE"
