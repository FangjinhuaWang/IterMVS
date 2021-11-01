#!/usr/bin/env bash

# train on DTU's training set
MVS_TRAINING="/home/Desktop/dtu/"

LOG_DIR="./checkpoints/dtu"

python train.py --dataset dtu_yao --batch_size 4 --epochs 1 --lr 0.001 --lrepochs 4,8,12:2 \
--small_image --iteration 4 \
--trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --vallist lists/dtu/val.txt \
--logdir=$LOG_DIR $@

python train.py --dataset dtu_yao --batch_size 4 --epochs 16 --lr 0.001 --lrepochs 4,8,12:2 --regress --resume \
--small_image --iteration 4 \
--trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --vallist lists/dtu/val.txt \
--logdir=$LOG_DIR $@
