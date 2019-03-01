#!/bin/bash

export PYTHONPATH="$(pwd)"

python3 main.py \
  --model_name="naive" \
  --reset_output_dir \
  --data_path="." \
  --output_dir="outputs" \
  --n_classes=50 \
  --n_epochs=50 \
  --train_steps=10000 \
  --batch_size=32 \
  --log_every=1000 \
  "$@"

