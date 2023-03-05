#!/bin/bash
direct="zh_ar" # "ar_zh", "zh_ar"
device=6
test_data_path="/data5/wenqiao_data/nmt/CLAPS_NMT/processed_datasets"
test_batch_size=16
pretrained_model="/data5/wenqiao_data/nmt/CLAPS_NMT/pretrained_model/m2m100_1.2B"
read_model_path="/data5/wenqiao_data/nmt/CLAPS_NMT/saved_model/$direct"
res_dir="/data5/wenqiao_data/nmt/CLAPS_NMT/result"
max_length=64
max_decode_step=64
beam_size=15

echo "-----------------------------------------------------generate ${direct}-----------------------------------------------------"
python run.py --direct $direct --testing --device $device --test_data_path $test_data_path \
        --test_batch_size $test_batch_size --pretrained_model $pretrained_model --read_model_path $read_model_path\
        --res_dir $res_dir --max_length $max_length --max_decode_step $max_decode_step --beam_size $beam_size

echo "${direct} finished!"
