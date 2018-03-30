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

n_epochs=5
batch_size=5
train_sim_num=40
test_sim_num=5
sim_len=100
input_len=5
output_len=1
save_dir="../saves/$start_time"

mkdir -p $save_dir
python main.py --train-sim-num=$train_sim_num --test-sim-num=$test_sim_num --sim-len=$sim_len \
--batch-size=$batch_size --input-len=$input_len --output-len=$output_len --save-dir=$save_dir \
--n-epochs=$n_epochs

