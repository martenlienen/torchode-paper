#!/bin/bash

python vdp.py --mu 25.0 --atol 1e-5 --rtol 1e-5 \
       --batch-size $(python -c "print(' '.join([str(2**i) for i in range(11)]))") \
       --warmup 0 --trials 1 --steps 2048 --out vdp-steps.pickle \
       --methods torchode torchdiffeq torchdyn diffrax
