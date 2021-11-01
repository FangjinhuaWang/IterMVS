#!/usr/bin/env bash

CKPT_FILE="./checkpoints/dtu/model_000015.ckpt"

DTU_TESTING="/home/Desktop/dtu/"

OUT_DIR="./outputs"

python eval.py --dataset=dtu_yao_eval --batch_size=1 --n_views 5 --iteration 4 \
--testpath=$DTU_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.3 \
--outdir=$OUT_DIR --testlist lists/dtu/test.txt --loadckpt $CKPT_FILE $@

