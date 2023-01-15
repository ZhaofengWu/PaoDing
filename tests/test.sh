#!/bin/bash

set -Exeuo pipefail

OUTPUT_DIR=tests/output/stsb

python tests/test_regression.py \
--data_dir data_cache \
--transformer_model bert-base-cased \
--batch_size 32 \
--max_length 256 \
--lr 0.00001 \
--warmup_ratio 0.06 \
--epochs 1 \
--clip_norm 1.0 \
--output_dir ${OUTPUT_DIR}

rm -rf ${OUTPUT_DIR}
