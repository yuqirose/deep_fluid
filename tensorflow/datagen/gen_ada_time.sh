#!/bin/bash

train_total=40
test_total=5
train_folder=../train_data/
test_folder=../test_data/

echo "simulate $train_total train and $test_total test."
counter=1
while [ $counter -le $train_total ]
do
    ../../build/manta genSimSingle.py basepath $train_folder savenpz 1 saveppm 1 
    echo "finish sim $counter train data."
done

counter=1
while [ $counter -le $test_total ]
do
    ../../build/manta genSimSingle.py basepath $test_folder savenpz 1 saveppm 1 
    echo "finish sim $counter test data."
done
