#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 \
python train.py -opt options/train/MSRS/train_lwavfu2.yml --debug