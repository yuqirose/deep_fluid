#!/bin/bash
timestamp() {
  date +"%T"
}
datestamp() {
  date +"%D"
}

t=$(timestamp)
t="$(echo ${t} | tr ':' '-')"

d=$(datestamp)
d=$(echo ${d} | tr '/' '-')

start_time="$d-$t"

train_sim_num=1
test_sim_num=1
sim_len=100

h_dim=1024

n_epochs=100
n_layers=3
batch_size=5
lr=1e-3
l2=1e-2

input_len=5
output_len=1
save_dir="../saves/$start_time"

mkdir -p $save_dir
python main.py --train-sim-num=$train_sim_num --test-sim-num=$test_sim_num --sim-len=$sim_len \
--h-dim=$h_dim --n-epochs=$n_epochs --n-layers=$n_layers --batch-size=$batch_size --lr=$lr --l2=$l2 \
--input-len=$input_len --output-len=$output_len --save-dir=$save_dir 

cp $(pwd)/run_medium.sh ${save_dir}/run_medium.sh

