#!/bin/sh
# Instruction-tune the converged d64/d96/d128 bases on QA data -> out_chat_d64/d96/d128.
cd "C:/Users/Mhlit/Desktop/Coding/TinyStories recreation/tinylm" || exit 1
PY=C:/Users/Mhlit/tinystories/.venv/Scripts/python.exe
DATA=rag_poc/qa512_train.jsonl
TOK=data/tok512.model

for M in d64 d96 d128; do
  SRC=out_width_${M}_long
  OUT=out_chat_${M}
  echo "[chat $M] fine-tune $SRC -> $OUT"
  $PY rag_poc/finetune_qa.py --ckpt_dir $SRC --tok $TOK --data $DATA --out_dir $OUT --steps 1500 \
      > data/train_chat_${M}.log 2>&1
  echo "[chat $M] exit=$? $(grep -E '^step 1500:' data/train_chat_${M}.log | tail -1)"
done

echo "CHAT_COMPARE_ALL_DONE"
