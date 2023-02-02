#!/bin/bash

python eval.py \
  -a resnet50 \
  --gpu 1 \
  --batch-size 64 \
  --mlp \
  --data-A 'datasets/OfficeHome/Product' \
  --data-B 'datasets/OfficeHome/Clipart' \
  --num-cluster '65' \
  --temperature 0.2 \
  --model 'office-home_product-clipart/model_best.pth.tar' \
  --cwcon-filterthresh 0.2 \
  --selfentro-temp 0.01 \
  --prec-nums '1,5,15' \
  

