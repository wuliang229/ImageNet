#!/bin/bash

export PYTHONPATH="$(pwd)"

python3 main.py \
  --model_name="naive" \
  --reset_output_dir \
  --data_path="." \
  --output_dir="outputs" \
  --n_classes=50 \
  --n_epochs=2 \
  --train_steps=700 \
  --batch_size=32 \
  --log_every=700 \
  "$@"


python3 main.py \
  --model_name="my_model" \
  --reset_output_dir \
  --data_path="." \
  --output_dir="outputs" \
  --n_classes=50 \
  --n_epochs=50 \
  --train_steps=700 \
  --batch_size=32 \
  --log_every=700 \
  "$@"

