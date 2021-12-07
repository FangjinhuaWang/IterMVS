#!/usr/bin/env bash

CKPT_FILE="./checkpoints/dtu/model_000015.ckpt"
# CKPT_FILE="./checkpoints/blendedmvs/model_000015.ckpt"

CUSTOM_TESTING="/home/Desktop/custom/"

OUT_DIR="./outputs"

python eval.py --dataset=custom --batch_size=1 --n_views 7 --iteration 4 --img_wh 640 480 \
--testpath=$CUSTOM_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.3 \
--outdir=$OUT_DIR --loadckpt $CKPT_FILE $@
