#!/bin/bash

for seed in 378328023 1778049253 538741903; do
    for lib in torchode torchode-joint torchdiffeq torchdyn; do
    python train_cnf.py --data mnist --dims 64,64,64 --strides 1,1,1,1 \
           --layer_type concat --multiscale False --parallel False \
           --autoencode False --rademacher True --conv True --train_T False \
           --batch_size 500 --num_epochs 25 --log_freq 1 --max_grad_norm 500 \
           --manual_seed "${seed}" --library "${lib}"
  done
done
