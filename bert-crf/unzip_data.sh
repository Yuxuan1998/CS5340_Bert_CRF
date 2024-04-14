#!/bin/bash

mkdir -p resources/data/train
mkdir -p resources/data/test

for train_part_id in $(seq 1 3); do
  unzip -d resources/data/train "resources/data/train_part_$train_part_id.zip"
done

unzip -d resources/data/test "resources/data/test_full.zip"