#!/bin/bash

# Run every method in its own process, so that they don't interfere in any way for
# accurate timings
for method in torchode-jit torchode torchdiffeq torchdyn diffrax-jit; do
  python vdp.py --mu 2.0 --atol 1e-5 --rtol 1e-5 \
         --batch-size 256 --warmup 2 --trials 3 --out vdp-benchmark.pickle \
         --steps 200 --methods $method
done
