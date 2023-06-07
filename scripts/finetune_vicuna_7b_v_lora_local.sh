#!/bin/bash
python /data/Im-sys/FastChat/fastchat/train/train_lora.py \
    --model_name_or_path /data/Im-sys/ckpts/vicuna-7b-v1.1  \
    --data_path /data/Im-sys/FastChat/fastchat/datasets/test_save.json \
    --bf16 False \
    --fp16 True \
    --output_dir /data/Im-sys/FastChat/output_7b_lora/ \
    --ckp_dir /data/Im-sys/FastChat/output_7b_ckp/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 10 \
    --evaluation_strategy "steps" \
    --eval_steps 10 \
    --save_strategy "steps" \
    --save_steps 10 \
    --save_total_limit 8 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --report_to none \

