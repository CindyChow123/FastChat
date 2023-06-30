# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

import sys

sys.path.append("/TTS_personal_jiahui.ni/Im-sys/FastChat/")

# Need to call this before importing transformers.
from fastchat.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

from fastchat.train.train import train

if __name__ == "__main__":
    # print(torch.cuda.is_available())
    # a = torch.cuda.device_count()
    # print(a)
    # for i in range(0,a):
    #     print(torch.cuda.get_device_name(i))
    train()
