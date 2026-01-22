#!/bin/bash

set -e

save_dir=../../../results_ANOLE
video_size_file_dir=../../../data/abr/video_sizes
val_trace_dir=../../../data/test
total_epoch=25000
train_trace_dir=../../../data/train
pretrained_model=../../../results_ANOLE/gener/model_saved/nn_model_ep_24500.ckpt
master_trace_path=../../../results_ANOLE/master_trace


seed=30
#for seed in 10 20 30; do
    python ../../simulator/ANOLE/train_master_gener.py  \
#done
