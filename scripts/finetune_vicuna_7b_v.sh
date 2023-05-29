#!/bin/bash
python /TTS_personal_jiahui.ni/Im-sys/FastChat/fastchat/train/train_deepspeed.py \
    --model_name_or_path /TTS_personal_jiahui.ni/Im-sys/ckpts/vicuna-7b-v1.1  \
    --data_path /TTS_personal_jiahui.ni/Im-sys/FastChat/fastchat/datasets/validation_prosocial_dialog_clean.json \
    --output_dir /TTS_personal_jiahui.ni/Im-sys/FastChat/output_7b \
    --ds_config_path /TTS_personal_jiahui.ni/Im-sys/FastChat/fastchat/train/vicuna-7b-deepseed.json \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 1500 \
    --save_strategy "steps" \
    --save_steps 1500 \
    --save_total_limit 8 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \

