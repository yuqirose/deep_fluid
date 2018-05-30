#!/bin/bash

python gan/train.py \
--trial 101 \
--model ConvLSTM \
--h_dim 256 \
--rnn_dim 128 \
--n_layers 2 \
--input_len 10 \
--output_len 5 \
--clip 10 \
--pre_start_lr 1e-3 \
--pre_min_lr 1e-4 \
--batch_size 32 \
--pretrain 30 \
--subsample 16 \
--discrim_rnn_dim 10 \
--discrim_layers 1 \
--policy_learning_rate 1e-5 \
--discrim_learning_rate 1e-4 \
--pretrain_disc_iter 2000 \
--max_iter_num 1000 \
--log_freq 64 \
--plot_freq 256 \
--train_dir /cs/ml/datasets/smoke/train_data \
--test_dir /cs/ml/datasets/smoke/test_data \
--cuda
# --cont
