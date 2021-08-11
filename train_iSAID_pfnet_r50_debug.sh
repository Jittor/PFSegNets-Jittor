#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./iSAID/
mkdir -p ${EXP_DIR}
CUDA_VISIBLE_DEVICES=0 python train.py \
  --dataset iSAID \
  --cv 0 \
  --arch network.pointflow_resnet_with_max_avg_pool.DeepR50_PF_maxavg_deeply \
  --class_uniform_tile 1024 \
  --max_cu_epoch 16 \
  --lr 0.007 \
  --lr_schedule poly \
  --poly_exp 0.9 \
  --repoly 1.5  \
  --rescale 1.0 \
  --sgd \
  --aux \
  --maxpool_size 14 \
  --avgpool_size 9 \
  --edge_points 128 \
  --match_dim 64 \
  --joint_edge_loss_pfnet \
  --edge_weight 25.0 \
  --ohem \
  --crop_size 896 \
  --max_epoch 16 \
  --wt_bound 1.0 \
  --bs_mult 4 \
  --exp r50_debug \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  --print_freq 1 \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt
