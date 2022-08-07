#!/bin/bash

DATASET_DIR="datasets/ICDAR"

mkdir -p $DATASET_DIR

wget https://lanytek.com/datasets/ICDAR_2015.tar

tar -xf ICDAR_2015.tar -C $DATASET_DIR

rm ICDAR_2015.tar