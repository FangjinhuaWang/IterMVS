#!/usr/bin/env bash

# train on BlendedMVS
MVS_TRAINING="/home/Desktop/BlendedMVS/"

LOG_DIR="./checkpoints/blendedmvs"

python train.py --dataset blendedmvs --batch_size 2 --epochs 1 --lr 0.001 --lrepochs 4,8,12:2 \
--iteration 4 \
--trainpath=$MVS_TRAINING1 --trainlist lists/blendedmvs/train.txt --vallist lists/blendedmvs/val.txt \
--logdir=$LOG_DIR $@

python train.py --dataset blendedmvs --batch_size 2 --epochs 16 --lr 0.001 --lrepochs 4,8,12:2 --regress --resume \
--iteration 4 \
--trainpath=$MVS_TRAINING1 --trainlist lists/blendedmvs/train.txt --vallist lists/blendedmvs/val.txt \
--logdir=$LOG_DIR $@
