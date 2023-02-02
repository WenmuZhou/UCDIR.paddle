#!/bin/bash
python eval.py \
  -a resnet50 \
  --gpu 6 \
  --batch-size 1 \
  --mlp \
  --data-A 'datasets/full/domainNet/clipart' \
  --data-B 'datasets/full/domainNet/sketch' \
  --num-cluster '7' \
  --temperature 0.2 \
  --model 'checkpoint_torch.params' \
  --cwcon-filterthresh 0.2 \
  --selfentro-temp 0.1 \
  --prec-nums '50,100,200'