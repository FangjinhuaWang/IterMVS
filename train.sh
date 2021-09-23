#!/usr/bin/env bash

# train on DTU's training set
# MVS_TRAINING="/local/home/fawang/Desktop/mvs_training/dtu/"
MVS_TRAINING="/cluster/scratch/fawang/mvs_training/dtu/"
MVS_TRAINING1="/cluster/scratch/fawang/BlendedMVS/"

# LOG_DIR="./test"
LOG_DIR="/cluster/scratch/fawang/mvs_320/checkpoints/9_22_349_3"
# LOAD_CKPT="./checkpoints/5_18/pretrain_2/model_000003.ckpt"
python train.py --dataset dtu_yao --batch_size 4 --epochs 1 --lr 0.001 --lrepochs 4,8,12:2 \
--small_image --iteration 4 \
--trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --vallist lists/dtu/test.txt \
--logdir=$LOG_DIR $@

python train.py --dataset dtu_yao --batch_size 4 --epochs 16 --lr 0.001 --lrepochs 4,8,12:2 --regress --resume \
--small_image --iteration 4 \
--trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --vallist lists/dtu/test.txt \
--logdir=$LOG_DIR $@

# python train.py --dataset blendedmvs --batch_size 4 --epochs 12 --lr 0.001 --lrepochs 4:10 --regress --resume \
# --iteration 4 \
# --trainpath=$MVS_TRAINING1 --trainlist lists/blendedmvs/train.txt --vallist lists/blendedmvs/val.txt \
# --logdir=$LOG_DIR $@
