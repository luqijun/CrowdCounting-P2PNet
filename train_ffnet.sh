#!/bin/bash

# 第一个参数
echo " ===================================训练arch: $1 ==================================="

CUDA_VISIBLE_DEVICES=0 \
python train.py --data_root ./data/SHA \
    --dataset_file SHHA_New \
    --epochs 3500 \
    --lr_drop 3500 \
    --output_dir ./logs \
    --checkpoints_dir ./weights \
    --tensorboard_dir ./logs \
    --lr 0.0001 \
    --lr_backbone 0.00001 \
    --batch_size 8 \
    --arch $1 \
    --eval_start 150 \
    --eval_freq 1 \
    --gpu_id 0 #\
#    --resume logs/0726-135150/latest.pth