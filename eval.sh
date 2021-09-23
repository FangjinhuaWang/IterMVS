#!/usr/bin/env bash


CKPT_FILE="/cluster/scratch/fawang/mvs_320/checkpoints/8_17_331_61/model_000011.ckpt"
#CKPT_FILE="./6_20_331_43/model_000011.ckpt"

DTU_TESTING="/cluster/scratch/fawang/dtu/"
ETH3D_TESTING="/cluster/scratch/fawang/eth3d_high_res_test/"
TANK_TESTING="/cluster/scratch/fawang/tankandtemples/"

OUT_DIR="/cluster/scratch/fawang/mvs_320/outputs_331_61"
#OUT_DIR="./outputs"

python eval_old.py --dataset=dtu_yao_eval --batch_size=1 --n_views 5 --iteration 4 \
--testpath=$DTU_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.5 \
--outdir=$OUT_DIR --testlist lists/dtu/test.txt --loadckpt $CKPT_FILE $@

python eval_old.py --dataset=tanks --split intermediate --batch_size=1 --n_views 7 iteration 4 \
--testpath=$TANK_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.3 \
--outdir=$OUT_DIR --loadckpt $CKPT_FILE $@

python eval_old.py --dataset=tanks --split advanced --batch_size=1 --n_views 7 --iteration 4 \
--testpath=$TANK_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.3 \
--outdir=$OUT_DIR --loadckpt $CKPT_FILE $@

python eval_old.py --dataset=eth3d --split train --batch_size=1 --n_views 7 --iteration 4 \
--testpath=$ETH3D_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.3 \
--outdir=$OUT_DIR --loadckpt $CKPT_FILE $@

python eval_old.py --dataset=eth3d --split test --batch_size=1 --n_views 7 --iteration 4 \
--testpath=$ETH3D_TESTING --geo_pixel_thres=1 --geo_depth_thres=0.01 --photo_thres=0.3 \
--outdir=$OUT_DIR --loadckpt $CKPT_FILE $@