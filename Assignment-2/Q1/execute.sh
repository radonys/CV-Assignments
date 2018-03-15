#!/bin/bash
arr=( 10 20 30 40 50 60 70 80 90 100 )
for item in ${arr[*]}
do
y="python Q1.py --mode=4 --confusion=1 --test_label_path=../../features/labels_test_features$item.pkl --train_label_path=../../features/labels_training_features_$item.pkl --train_data_path=../../features/training_features$item.pkl --test_data_path=../../features/test_features$item.pkl --k=5 &"
eval $y
done
