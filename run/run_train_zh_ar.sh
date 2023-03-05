#!/bin/bash
model_name="m2m" # "mbart", "m2m"
direct="zh_ar" # "ar_zh", "zh_ar"
seed=2022
device=2
train_data_path="/data5/wenqiao_data/nmt/CLAPS_NMT/processed_datasets"
dev_data_path="/data5/wenqiao_data/nmt/CLAPS_NMT/processed_datasets"
train_batch_size=16
grad_accumulate=4
dev_batch_size=8
pretrained_model="/data5/wenqiao_data/nmt/CLAPS_NMT/pretrained_model/m2m100_1.2B"
read_model_path=""
max_length=64
max_decode_step=64
beam_size=10
warmup_ratio=0.1
lr=5e-5
l2=0.1
max_norm=5
lr_schedule=linear
num_epoch=25
eval_after_epoch=1

# contrastive learnng
tau=0.1
pos_eps=3.0
neg_eps=3.0
alpha=10
beta=1

echo "-----------------------------------------------------training ${direct}-----------------------------------------------------"
python -u run.py --direct $direct $testing --seed $seed --device $device --train_data_path $train_data_path \
        --dev_data_path $dev_data_path --train_batch_size $train_batch_size --dev_batch_size $dev_batch_size \
        --pretrained_model $pretrained_model $read_model_path --max_length $max_length --max_decode_step $max_decode_step \
        --num_epoch $num_epoch --warmup_ratio $warmup_ratio --lr $lr --l2 $l2 --max_norm $max_norm --lr_schedule $lr_schedule \
        --eval_after_epoch $eval_after_epoch --beam_size $beam_size --tau $tau --pos_eps $pos_eps --neg_eps $neg_eps \
        --alpha $alpha --beta $beta --model_name $model_name --grad_accumulate $grad_accumulate

echo "${direct} finished!"
