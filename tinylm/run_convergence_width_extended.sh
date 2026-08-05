#!/bin/sh
# Extended width sweep (d112/d160/d224) + iso-param shape points, run after the MoE e32 run.
# Width bridge d112/d160/d224 extends the scaling fit toward the released baselines;
# shape fill d56L7/d72L4 confirms the iso-param U. Then rebuild the table + regenerate figures.
cd "C:/Users/Mhlit/Desktop/Coding/TinyStories recreation/tinylm" || exit 1
PY=C:/Users/Mhlit/tinystories/.venv/Scripts/python.exe
PLOT=C:/Users/Mhlit/AppData/Local/Programs/Python/Python313/python.exe
COMMON="--vocab_source=custom --vocab_size=512 --batch_size=128 --max_seq_len=256 --max_iters=30000 --eval_interval=1000 --compile=False"

echo "[wait] waiting for MoE e32 (out_moe_d64_e32) to reach step 30000..."
i=0
while ! grep -qE '^step 30000: ' data/train_moe_e32.log 2>/dev/null; do
  sleep 120; i=$((i + 1))
  if [ $i -gt 180 ]; then echo "[wait] timed out after 6h; aborting"; exit 1; fi
done
echo "[wait] MoE e32 done; pausing 120s for checkpoint flush"; sleep 120

# --- width bridge: n_heads=8, n_kv_heads=4, 5 layers (matches the width sweep) ---
for DIM in 112 160 224; do
  OUT=out_width_d${DIM}_long; LOG=data/train_width_d${DIM}_long.log
  echo "[width d${DIM}] start -> $OUT"
  $PY train.py $COMMON --n_heads=8 --n_kv_heads=4 --n_layers=5 --dim=${DIM} --out_dir=$OUT > $LOG 2>&1
  echo "[width d${DIM}] exit=$? $(grep -E '^step 30000: ' $LOG | tail -1)"
done

# --- shape fill: iso-param ~279K; dim 56/72 use n_heads=4 for an even head_dim ---
echo "[shape d72L4] start"
$PY train.py $COMMON --n_heads=4 --n_kv_heads=2 --dim=72 --n_layers=4 --out_dir=out_iso_d72L4_long > data/train_iso_d72L4_long.log 2>&1
echo "[shape d72L4] exit=$? $(grep -E '^step 30000: ' data/train_iso_d72L4_long.log | tail -1)"
echo "[shape d56L7] start"
$PY train.py $COMMON --n_heads=4 --n_kv_heads=2 --dim=56 --n_layers=7 --out_dir=out_iso_d56L7_long > data/train_iso_d56L7_long.log 2>&1
echo "[shape d56L7] exit=$? $(grep -E '^step 30000: ' data/train_iso_d56L7_long.log | tail -1)"

# --- rebuild the master table (torch venv) + regenerate figures (matplotlib) ---
echo "[rebuild] master_data.csv"
$PY build_master.py > data/build_master_rebuild.log 2>&1
echo "[rebuild] figures"
$PLOT ../reports/make_scaling_figure.py
$PLOT ../reports/make_ablation_plots.py

echo "WIDTH_EXTENDED_DONE"
