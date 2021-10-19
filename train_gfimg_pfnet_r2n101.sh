#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./GAOFENIMG/
mkdir -p ${EXP_DIR}
python train.py \
  --dataset GAOFENIMG \
  --cv 0 \
  --arch network.pointflow_resnet_with_max_avg_pool.DeepR2N101_PF_maxavg_deeply \
  --class_uniform_tile 1024 \
  --max_cu_epoch 64 \
  --lr 0.001 \
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
  --crop_size 512 \
  --max_epoch 64 \
  --wt_bound 1.0 \
  --bs_mult 8 \
  --exp r2n101 \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt
