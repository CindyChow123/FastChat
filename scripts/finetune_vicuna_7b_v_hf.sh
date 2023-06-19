#!/bin/bash
python /TTS_personal_jiahui.ni/Im-sys/FastChat/fastchat/train/train.py \
    --model_name_or_path /TTS_personal_jiahui.ni/Im-sys/ckpts/vicuna-7b-v1.1  \
    --data_path /TTS_personal_jiahui.ni/Im-sys/FastChat/fastchat/datasets/validation_prosocial_dialog_clean.json \
    --bf16 False \
    --output_dir /TTS_personal_jiahui.ni/Im-sys/FastChat/output_7b_fp16/ \
    --ckp_dir /TTS_personal_jiahui.ni/Im-sys/FastChat/output_7b_ckp/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 10 \
    --evaluation_strategy "steps" \
    --eval_steps 20 \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --deepspeed /TTS_personal_jiahui.ni/Im-sys/FastChat/fastchat/train/vicuna-deepseed-hf-noffl.json \
    --report_to none \

