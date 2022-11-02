#!/bin/bash

for seed in 941432720 2094771200 1591351728; do
    for lib in torchode torchode-jit torchdiffeq torchdyn; do
    python examples/train.py --library "${lib}" \
           --batch-size 8 --seed "${seed}" --max-grad-norm 10.0 \
           --group black-sea black-sea
  done
done
