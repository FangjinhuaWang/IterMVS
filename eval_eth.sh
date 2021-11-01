#!/usr/bin/env bash

CKPT_FILE="./checkpoints/dtu/model_000015.ckpt"
# CKPT_FILE="./checkpoints/blendedmvs/model_000015.ckpt"

ETH3D_TESTING="/home/Desktop/eth3d_high_res_test/"

OUT_DIR="./outputs"

python eval.py --dataset=eth3d --split train --batch_size=1 --n_views 7 --iteration 4 \
--testpath=$ETH3D_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.3 \
--outdir=$OUT_DIR --loadckpt $CKPT_FILE $@

python eval.py --dataset=eth3d --split test --batch_size=1 --n_views 7 --iteration 4 \
--testpath=$ETH3D_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.3 \
--outdir=$OUT_DIR --loadckpt $CKPT_FILE $@
