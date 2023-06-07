"""
Chat with a model with command line interface.

Usage:
python3 -m fastchat.serve.cli --model lmsys/fastchat-t5-3b-v1.0
python3 -m fastchat.serve.cli --model ~/model_weights/vicuna-7b
"""
import argparse
import os
import re

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live

import sys
sys.path.append("/TTS_personal_jiahui.ni/Im-sys/FastChat/")

from fastchat.model.model_adapter import add_model_args, get_global_rank
from fastchat.serve.inference import chat_loop, ChatIO

global_rank = None
def rank0_print(*arg,end:str="\n",flush:bool=False):
    """print the conversation if the global rank is 0, for multi host application"""
    global global_rank
    if(global_rank is None):
        global_rank = get_global_rank()
    if(global_rank == 0):
        with open(args.conv_out_path,"a") as f:
            print(*arg, end = end, flush = flush, file=f)

class FileChatIO(ChatIO):
    inputs = None
    input_index = -1
    def __init__(self, file_path: str) -> None:
        super().__init__()
        with open(file_path,"r") as f:
            self.inputs = f.readlines()

    def prompt_for_input(self, role: str) -> str:
        if(self.input_index < len(self.inputs)-1):
            self.input_index += 1
            rank0_print(f"{role}: ", self.inputs[self.input_index].strip('\n'))
            return self.inputs[self.input_index]
        else:
            return None
        
    def prompt_for_output(self, role: str):
        rank0_print(f"{role}: ", end="", flush=True)
        
    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                rank0_print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        rank0_print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)

class SimpleChatIO(ChatIO):
    def prompt_for_input(self, role) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str):
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream):
        pre = 0
        for outputs in output_stream:
            output_text = outputs["text"]
            output_text = output_text.strip().split(" ")
            now = len(output_text) - 1
            if now > pre:
                print(" ".join(output_text[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(output_text[pre:]), flush=True)
        return " ".join(output_text)


class RichChatIO(ChatIO):
    def __init__(self):
        self._prompt_session = PromptSession(history=InMemoryHistory())
        self._completer = WordCompleter(
            words=["!exit", "!reset"], pattern=re.compile("$")
        )
        self._console = Console()

    def prompt_for_input(self, role) -> str:
        self._console.print(f"[bold]{role}:")
        # TODO(suquark): multiline input has some issues. fix it later.
        prompt_input = self._prompt_session.prompt(
            completer=self._completer,
            multiline=False,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=None,
        )
        self._console.print()
        return prompt_input

    def prompt_for_output(self, role: str):
        self._console.print(f"[bold]{role}:")

    def stream_output(self, output_stream):
        """Stream output from a role."""
        # TODO(suquark): the console flickers when there is a code block
        #  above it. We need to cut off "live" when a code block is done.

        # Create a Live context for updating the console output
        with Live(console=self._console, refresh_per_second=4) as live:
            # Read lines from the stream
            for outputs in output_stream:
                if not outputs:
                    continue
                text = outputs["text"]
                # Render the accumulated text as Markdown
                # NOTE: this is a workaround for the rendering "unstandard markdown"
                #  in rich. The chatbots output treat "\n" as a new line for
                #  better compatibility with real-world text. However, rendering
                #  in markdown would break the format. It is because standard markdown
                #  treat a single "\n" in normal text as a space.
                #  Our workaround is adding two spaces at the end of each line.
                #  This is not a perfect solution, as it would
                #  introduce trailing spaces (only) in code block, but it works well
                #  especially for console output, because in general the console does not
                #  care about trailing spaces.
                lines = []
                for line in text.splitlines():
                    lines.append(line)
                    if line.startswith("```"):
                        # Code block marker - do not add trailing spaces, as it would
                        #  break the syntax highlighting
                        lines.append("\n")
                    else:
                        lines.append("  \n")
                markdown = Markdown("".join(lines))
                # Update the Live console output
                live.update(markdown)
        self._console.print()
        return text


def main(args):
    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    if args.style == "simple":
        chatio = SimpleChatIO()
    elif args.style == "rich":
        chatio = RichChatIO()
    elif args.style == "file":
        chatio = FileChatIO(args.conv_file)
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    try:
        chat_loop(
            args.model_path,
            args.device,
            args.num_gpus,
            args.max_gpu_memory,
            args.load_8bit,
            args.cpu_offloading,
            args.conv_template,
            args.temperature,
            args.max_new_tokens,
            chatio,
            args.debug,
            args.use_deepspeed,
            args.lora_path
        )
    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--style",
        type=str,
        default="simple",
        choices=["simple", "rich", "file"],
        help="Display style.",
    )
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    parser.add_argument("--conv-file", type=str, default=None, help="User prompt file for conversation under file chat mode")
    parser.add_argument("--use-deepspeed", action="store_true", help="Whether using deepspeed to accelerate")
    parser.add_argument("--conv-out-path", type=str, default=None, help="The output file path for generated conversation")
    parser.add_argument("--lora-path", type=str, default=None,help="Lora weight directory")
    args = parser.parse_args()
    main(args)
