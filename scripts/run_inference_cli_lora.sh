#!/bin/bash
# python3 /data/Im-sys/FastChat/fastchat/serve/controller.py; python3 /data/Im-sys/FastChat/fastchat/serve/model_worker.py --model-path /data/Im-sys/ckpts/vicuna-13b-v1.1 --load-8bit --cpu-offloading; python3 /data/Im-sys/FastChat/fastchat/serve/gradio_web_server.py --share
python3 /TTS_personal_jiahui.ni/Im-sys/FastChat/fastchat/serve/cli.py \
    --model-path /TTS_personal_jiahui.ni/Im-sys/ckpts/vicuna-7b-v1.1 \
    --num-gpus 1 \
    --style file \
    --conv-file /TTS_personal_jiahui.ni/Im-sys/FastChat/fastchat/datasets/Conv_example \
    --conv-out-path /TTS_personal_jiahui.ni/logs/conv_out.log \
    --lora-path /TTS_personal_jiahui.ni/Im-sys/FastChat/output_7b_lora