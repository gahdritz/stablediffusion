#!/bin/bash

module load cuda cudnn

python3 robust_classification.py \
   outputs/txt2img-samples/astronaut_horse.png \
   configs/stable-diffusion/v2-inference-v.yaml \
   ckpts/v2-1_768-ema-pruned.ckpt \
   simple_class_file.txt \
   --batch_size 1 \
   --use_pickle_cache \
   --pickle_cache_dir pickles
