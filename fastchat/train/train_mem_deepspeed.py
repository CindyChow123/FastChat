# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

import sys
sys.path.append("/TTS_personal_jiahui.ni/Im-sys/FastChat/")

# Need to call this before importing transformers.
from fastchat.train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

from fastchat.train.train_deepspeed import main

if __name__ == "__main__":
    main()
