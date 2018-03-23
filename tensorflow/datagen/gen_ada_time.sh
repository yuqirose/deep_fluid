#!/bin/bash

total=80
counter=1
echo "simulate $total times."
while [ $counter -le $total ]
do
    ../../build/manta genSimSingle.py saveuni 1 saveppm 1 
    echo "finish sim $counter."
done
