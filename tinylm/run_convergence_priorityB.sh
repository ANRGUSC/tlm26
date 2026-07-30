#!/bin/sh
# Priority B: retrain tokenizer, quantization, and data-quantity families to 30k -> out_*_long.
cd "C:/Users/Mhlit/Desktop/Coding/TinyStories recreation/tinylm" || exit 1
PY=C:/Users/Mhlit/tinystories/.venv/Scripts/python.exe
COMMON="--vocab_source=custom --n_heads=8 --n_kv_heads=4 --batch_size=128 --max_seq_len=256 --max_iters=30000 --eval_interval=1000 --compile=False --n_layers=5"

fin () { echo "[$1] exit=$2 $(grep -E '^step 30000: ' data/$3.log | tail -1)"; }

# ---- TOKENIZER family (dim 64) ----
echo "[char105] start";   $PY train.py $COMMON --dim=64 --vocab_size=105  --out_dir=out_char105_long   > data/out_char105_long.log 2>&1;   fin char105 $? out_char105_long
echo "[tok256] start";    $PY train.py $COMMON --dim=64 --vocab_size=256  --out_dir=out_tok_256_long   > data/out_tok_256_long.log 2>&1;   fin tok256 $? out_tok_256_long
echo "[tok512dep] start"; TS_TOK_NAME=tok512dep $PY train.py $COMMON --dim=64 --vocab_size=512 --out_dir=out_tok_512dep_long > data/out_tok_512dep_long.log 2>&1; fin tok512dep $? out_tok_512dep_long
echo "[tok512nf] start";  TS_TOK_NAME=tok512nf  $PY train.py $COMMON --dim=64 --vocab_size=512 --out_dir=out_tok_512nf_long  > data/out_tok_512nf_long.log 2>&1;  fin tok512nf $? out_tok_512nf_long
echo "[tok512uni] start"; TS_TOK_NAME=tok512uni $PY train.py $COMMON --dim=64 --vocab_size=512 --out_dir=out_tok_512uni_long > data/out_tok_512uni_long.log 2>&1; fin tok512uni $? out_tok_512uni_long
echo "[B/1024] start";    $PY train.py $COMMON --dim=64 --vocab_size=1024 --out_dir=out_B_long        > data/out_B_long.log 2>&1;         fin B $? out_B_long
echo "[C/2048] start";    $PY train.py $COMMON --dim=64 --vocab_size=2048 --out_dir=out_C_long        > data/out_C_long.log 2>&1;         fin C $? out_C_long

# ---- QUANTIZATION family (tok512) ----
echo "[qat_int4] start";       $PY train.py $COMMON --dim=64 --vocab_size=512 --qat_level=int4    --out_dir=out_qat_int4_long       > data/out_qat_int4_long.log 2>&1;       fin qat_int4 $? out_qat_int4_long
echo "[qat_ternary] start";    $PY train.py $COMMON --dim=64 --vocab_size=512 --qat_level=ternary --out_dir=out_qat_ternary_long    > data/out_qat_ternary_long.log 2>&1;    fin qat_ternary $? out_qat_ternary_long
echo "[qat_ternary_d96] start";$PY train.py $COMMON --dim=96 --vocab_size=512 --qat_level=ternary --out_dir=out_qat_ternary_d96_long > data/out_qat_ternary_d96_long.log 2>&1; fin qat_ternary_d96 $? out_qat_ternary_d96_long

# ---- DATA-QUANTITY family (tok512, vary train tokens) ----
echo "[t300k] start"; TS_MAX_TRAIN_TOKENS=300000   $PY train.py $COMMON --dim=64 --vocab_size=512 --out_dir=out_data_t300k_long > data/out_data_t300k_long.log 2>&1; fin t300k $? out_data_t300k_long
echo "[t1m] start";   TS_MAX_TRAIN_TOKENS=1000000  $PY train.py $COMMON --dim=64 --vocab_size=512 --out_dir=out_data_t1m_long   > data/out_data_t1m_long.log 2>&1;   fin t1m $? out_data_t1m_long
echo "[t3m] start";   TS_MAX_TRAIN_TOKENS=3000000  $PY train.py $COMMON --dim=64 --vocab_size=512 --out_dir=out_data_t3m_long   > data/out_data_t3m_long.log 2>&1;   fin t3m $? out_data_t3m_long
echo "[t10m] start";  TS_MAX_TRAIN_TOKENS=10000000 $PY train.py $COMMON --dim=64 --vocab_size=512 --out_dir=out_data_t10m_long > data/out_data_t10m_long.log 2>&1; fin t10m $? out_data_t10m_long

echo "PRIORITY_B_ALL_DONE"
