#!/bin/sh
# Ternary-QAT rescue (d64/d96) + 64-expert MoE, from the pairing assessment (docs section 10).
# Runs sequentially on one GPU; the shorter ternary runs first, MoE-64e last.
#   1. ternary-rescue d64  (keep embeddings/output fp -> test if it fixes d64 ternary, was 1.21 bpb)
#   2. ternary-rescue d96  (was 1.025 bpb)
#   3. MoE 64e             (continue the expert-scaling trend 0.705 -> 0.658 -> 0.612 -> ?)
# Then eval bpb for everything and rebuild the data table + figures.
cd "C:/Users/Mhlit/Desktop/Coding/TinyStories recreation/tinylm" || exit 1
PY=C:/Users/Mhlit/tinystories/.venv/Scripts/python.exe
PLOT=C:/Users/Mhlit/AppData/Local/Programs/Python/Python313/python.exe
DENSE="--vocab_source=custom --vocab_size=512 --n_heads=8 --n_kv_heads=4 --batch_size=128 --max_seq_len=256 --max_iters=30000 --eval_interval=1000 --compile=False --n_layers=5"
LOG=data/ternary_rescue_moe64_results.log
: > $LOG
echo "=== experiment batch start ===" | tee -a $LOG

# 1. ternary-rescue d64
echo "[ternary_rescue_d64] start" | tee -a $LOG
$PY train.py $DENSE --dim=64 --qat_level=ternary --qat_skip_embed=True --out_dir=out_qat_ternary_rescue_d64_long > data/train_ternary_rescue_d64.log 2>&1
echo "[ternary_rescue_d64] exit=$? $(grep -E '^step 30000:' data/train_ternary_rescue_d64.log | tail -1)" | tee -a $LOG
$PY eval_moe.py out_qat_ternary_rescue_d64_long 2>/dev/null | grep -viE "PretokDataset|seed" | tee -a $LOG

# 2. ternary-rescue d96
echo "[ternary_rescue_d96] start" | tee -a $LOG
$PY train.py $DENSE --dim=96 --qat_level=ternary --qat_skip_embed=True --out_dir=out_qat_ternary_rescue_d96_long > data/train_ternary_rescue_d96.log 2>&1
echo "[ternary_rescue_d96] exit=$? $(grep -E '^step 30000:' data/train_ternary_rescue_d96.log | tail -1)" | tee -a $LOG
$PY eval_moe.py out_qat_ternary_rescue_d96_long 2>/dev/null | grep -viE "PretokDataset|seed" | tee -a $LOG

# 3. MoE 64e (long)
echo "[moe_e64] start" | tee -a $LOG
$PY train_moe.py --vocab_source=custom --vocab_size=512 --dim=64 --n_layers=5 --n_heads=8 --n_kv_heads=4 \
   --batch_size=128 --max_seq_len=256 --max_iters=30000 --eval_interval=1000 --compile=False \
   --moe_top_k=2 --n_experts=64 --out_dir=out_moe_d64_e64 > data/train_moe_e64.log 2>&1
echo "[moe_e64] exit=$? $(grep -E '^step 30000:' data/train_moe_e64.log | tail -1)" | tee -a $LOG

# 4. final eval + rebuild
echo "=== MoE expert-scaling trend (bpb) ===" | tee -a $LOG
$PY eval_moe.py out_moe_d64_e8 out_moe_d64_e16 out_moe_d64_e32 out_moe_d64_e64 2>/dev/null | grep -viE "PretokDataset|seed" | tee -a $LOG
echo "=== ternary-rescue vs old ternary (from master) ===" | tee -a $LOG
$PY build_master.py > data/build_master_rebuild.log 2>&1
grep -E "ternary" ../reports/master_data.csv | tee -a $LOG
$PLOT ../reports/make_scaling_figure.py >> data/build_master_rebuild.log 2>&1
$PLOT ../reports/make_ablation_plots.py >> data/build_master_rebuild.log 2>&1
echo "EXPERIMENTS_DONE" | tee -a $LOG
