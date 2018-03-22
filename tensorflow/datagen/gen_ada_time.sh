#!/bin/bash
count = 100
for i in $count; do
	python genSimSingle.py saveuni 1 
    echo $i
done
