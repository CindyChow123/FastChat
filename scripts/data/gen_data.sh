#!/bin/bash
python /TTS_personal_jiahui.ni/Im-sys/FastChat/fastchat/data/gen_revChatGPT.py \
    --revgpt \
    --log \
    --log-file /TTS_personal_jiahui.ni/Im-sys/FastChat/fastchat/data/host_rev_add_davinci_0626.log \
    --output-file /TTS_personal_jiahui.ni/Im-sys/FastChat/fastchat/datasets/RoboEmo/Host_rev_add_davinci_0626 \
    --error-file /TTS_personal_jiahui.ni/logs/data_0625.log \
    --base-dir /TTS_personal_jiahui.ni/