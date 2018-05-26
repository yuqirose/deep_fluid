#!/bin/bash

python gan/train.py \
--trial 101 \
--model ConvLSTM \
--h_dim 200 \
--rnn_dim 10 \
--n_layers 2 \
--clip 10 \
--pre_start_lr 1e-3 \
--pre_min_lr 1e-4 \
--batch_size 64 \
--pretrain 30 \
--subsample 16 \
--discrim_rnn_dim 128 \
--discrim_layers 1 \
--policy_learning_rate 1e-5 \
--discrim_learning_rate 1e-3 \
--pretrain_disc_iter 2000 \
--max_iter_num 60000 \
--cuda
# --cont
