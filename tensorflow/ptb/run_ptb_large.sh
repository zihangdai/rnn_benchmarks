#!/bin/bash -e
CUDA_VISIBLE_DEVICES=1 python ptb_word_lm.py --data_path=${HOME}/SSD/rnn_benchmark/torch/ptb/lstm/data --model large
