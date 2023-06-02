#!/bin/bash
# python3 /data/Im-sys/FastChat/fastchat/serve/controller.py; python3 /data/Im-sys/FastChat/fastchat/serve/model_worker.py --model-path /data/Im-sys/ckpts/vicuna-13b-v1.1 --load-8bit --cpu-offloading; python3 /data/Im-sys/FastChat/fastchat/serve/gradio_web_server.py --share
python3 /TTS_personal_jiahui.ni/Im-sys/FastChat/fastchat/serve/cli.py \
    --model-path /TTS_personal_jiahui.ni/Im-sys/ckpts/vicuna-13b-v1.1 \
    --use-deepspeed \
    --num-gpus 2 \
    --style file \
    --conv-file /TTS_personal_jiahui.ni/Im-sys/FastChat/fastchat/datasets/Conv_example \
    --conv-out-path /TTS_personal_jiahui.ni/logs/conv_out.log