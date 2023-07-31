#!/bin/bash
python /TTS_personal_jiahui.ni/Im-sys/FastChat/fastchat/train/train_lora.py \
    --model_name_or_path /TTS_team_data_01/xinyi.zhou/LLaMA_65B/ \
    --data_path /TTS_personal_jiahui.ni/Im-sys/FastChat/fastchat/datasets/RoboEmo/Host_16719.json \
    --bf16 False \
    --fp16 True \
    --output_dir /TTS_team_data_01/xinyi.zhou/LLaMA_65B_lora/ \
    --ckp_dir /TTS_team_data_01/xinyi.zhou/LLaMA_65B_ckp/ \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "steps" \
    --eval_steps 10 \
    --save_strategy "steps" \
    --save_steps 1 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --lazy_preprocess True \
    --report_to wandb \
    --deepspeed /TTS_personal_jiahui.ni/Im-sys/FastChat/fastchat/train/vicuna-deepseed-hf.json \
