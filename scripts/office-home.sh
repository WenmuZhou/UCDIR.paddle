#!/bin/bash

python main.py \
  -a resnet50 \
  --gpu 1 \
  --batch-size 64 \
  --mlp --aug-plus --cos \
  --data-A 'datasets/OfficeHome/Product' \
  --data-B 'datasets/OfficeHome/Clipart' \
  --num-cluster '65' \
  --warmup-epoch 20 \
  --temperature 0.2 \
  --exp-dir 'office-home_product-clipart' \
  --lr 0.0002 \
  --clean-model 'moco_v2_800ep_pretrain.params' \
  --model 'office-home_product-clipart/model_best.pth.tar' \
  --instcon-weight 1.0 \
  --cwcon-startepoch 20 \
  --cwcon-satureepoch 100 \
  --cwcon-weightstart 0.0 \
  --cwcon-weightsature 0.5 \
  --cwcon-filterthresh 0.2 \
  --epochs 200 \
  --selfentro-temp 0.01 \
  --selfentro-weight 1.0 \
  --selfentro-startepoch 100 \
  --distofdist-weight 0.5 \
  --distofdist-startepoch 100 \
  --prec-nums '1,5,15' \
  

