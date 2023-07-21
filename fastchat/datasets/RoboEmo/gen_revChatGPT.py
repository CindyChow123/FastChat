import asyncio, dataclasses, requests, openai, json, time, argparse, os, httpx
from typing import List, Tuple, Any, Dict, Set, Optional

from revChatGPT.V1 import Chatbot, AsyncChatbot

"""
python gen_revChatGPT.py >> host_0614_1136.log 2>&1"""



openai.api_key = ""


@dataclasses.dataclass
class Prompt:
    """A class for generating promp pattern"""

    main_prompt: str
    guest_type: List[str]
    total_words_limit: List[int]
    per_response_limit: List[int]
    specialty: str
    req_options: List[str]
    gt_init: int = 0
    total_init: int = 0
    per_init: int = 0
    opt_init: int = 0

    type_pre: str = "\nGuest type: "
    total_pre: str = "\nTotal words limit: "
    per_pre: str = "\nMinimum words per response: "
    special_pre: str = "\nSpecial requirements: "
    # who_to_ask: str="\nWho to ask: Guest"
    # who_to_answer: str="\nWho to answer: Robo"
    req_pre: str = "\nCurrent request: in the "

    def get_prompts(self):
        strr = self.main_prompt
        i = self.gt_init
        while i < len(self.guest_type):
            j = self.total_init
            while j < len(self.total_words_limit):
                k = self.per_init
                while k < len(self.per_response_limit):
                    m = self.opt_init
                    while m < len(self.req_options):
                        guest = self.type_pre + self.guest_type[i]
                        total = self.total_pre + str(self.total_words_limit[j])
                        per = self.per_pre + str(self.per_response_limit[k])
                        special = self.special_pre + self.specialty
                        req = self.req_pre + self.req_options[m]
                        yield strr + guest + total + per + special + req, f"gt-{i}-total-{j}-per-{k}-opt-{m}"
                        m += 1
                    k += 1
                j += 1
            i += 1


@dataclasses.dataclass
class ChatLoop:
    """A class for chatgpt chatting"""

    output_path: str
    log_path: str
    error_path: str
    conversation_id: List[str]
    access_token: List[str]  # ?? using List + field?
    begin_cnt: int = 1
    openai_model: str = "gpt-3.5-turbo-16k"
    chatbots: List = None
    chatbot_num: int = 0
    checker: int = 0

    async def async_write_response(self, prompt, extra, cnt):
        with open(self.error_path, "a") as err:
            with open(self.log_path, "a") as log:
                with open(self.output_path, "a") as f:
                    prev_text = ""
                    print(f"No {cnt}", end="\n", flush=True, file=f)
                    async for data in self.chatbots[self.chatbot_num].ask(prompt):
                        if self.conversation_id[self.chatbot_num] is None:
                            self.conversation_id[self.chatbot_num] = data[
                                "conversation_id"
                            ]
                            self.chatbots[
                                self.chatbot_num
                            ].conversation_id = self.conversation_id[self.chatbot_num]
                            print(
                                f"Conv id {self.conversation_id[self.chatbot_num]}",
                                file=err,
                                flush=True,
                            )
                        message = data["message"][len(prev_text) :]

                        if message.endswith("END"):
                            print(message, end="\n", flush=True, file=f)
                        else:
                            print(message, end="", flush=True, file=f)

                        prev_text = data["message"]
                    print(f"No.{cnt}: {extra}", flush=True, file=log)

    async def delete_conv(self):
        with open(self.error_path, "a") as err:
            await chat.chatbots[chat.chatbot_num].delete_conversation(
                chat.conversation_id[chat.chatbot_num]
            )
            self.conversation_id[self.chatbot_num] = None
            self.chatbots[self.chatbot_num].conversation_id = None
            print("Conversation deleted!", flush=True, file=err)

    async def switch_bot(self):
        self.checker = 0
        self.chatbot_num = 1 - self.chatbot_num

    async def start_async_chat(self, prompt_object: Prompt, logs: Set[str]):
        chatbot_gmail = AsyncChatbot(
            config={
                "access_token": self.access_token[0],
                "conversation_id": self.conversation_id[0],
                # "model": "gpt-4"
            }
        )
        chatbot_usc = AsyncChatbot(
            config={
                "access_token": self.access_token[1],
                "conversation_id": self.conversation_id[1],
                # "model": "gpt-4"
            }
        )
        chatbot_shitao = AsyncChatbot(
            config={
                "access_token": self.access_token[0],
                "conversation_id": self.conversation_id[0],
                # "model": "gpt-4"
            }
        )
        self.chatbots = [chatbot_shitao, chatbot_usc]

        # with open(self.log_path,'a') as log:
        # with open(self.output_path,'a') as f:
        with open(self.error_path, "a") as err:
            cnt = self.begin_cnt
            for prompt, extra in prompt_object.get_prompts():
                if extra not in logs:
                    try:
                        print(f"{self.chatbot_num}: {cnt}", flush=True, file=err)
                        await self.async_write_response(prompt, extra, cnt)
                        cnt += 1
                        self.checker += 1
                        if self.checker % 50 == 49:
                            await self.switch_bot()
                    except Exception as exc:
                        print(f"Error while requesting {exc}!", flush=True, file=err)
                        if self.conversation_id[self.chatbot_num] is not None:
                            await self.delete_conv()
                        await self.switch_bot()
                        await self.async_write_response(prompt, extra, cnt)
                        cnt += 1
                        continue
                    time.sleep(20)

    def start_openai_chat(self, prompt_object: Prompt, logs: Set[str]):
        with open(self.output_path, "a") as f:
            cnt = self.begin_cnt
            for prompt, extra in prompt_object.get_prompts():
                if extra not in logs:
                    print(f"No.{cnt}: {extra}", flush=True)
                    # print("Conversation:",file=f)
                    # create a chat completion
                    chat_completion = openai.ChatCompletion.create(
                        model=self.openai_model,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    # # print the chat completion
                    print(
                        chat_completion.choices[0].message.content, flush=True, file=f
                    )
                    cnt += 1
                    time.sleep(20)


if __name__ == "__main__":
    local = "/data/"

    openai_output = "Host"
    openai_log = "host_rev_add_gpt4.log"
    rev_output = "./temp/Host_rev_add_davinci_0713"
    rev_log = "./temp/Host_rev_add_davinci_0713.log"
    err_log = "./temp/Host_rev_add_davinci_0713_out.log"
    params_path = "./config/params.json"
    private_ids_json = "./config/private_ids.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=str, default=rev_output)
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--log-file", type=str, default=rev_log)
    parser.add_argument("--error-file", type=str, default=err_log)
    parser.add_argument("--openai", action="store_true", default=False)
    parser.add_argument("--revgpt", action="store_true", default=False)
    parser.add_argument("--main-prompt-name", type=str,default="recp_main_prompt")
    parser.add_argument("--params-json-path",type=str,default=params_path)
    parser.add_argument("--ids-json-path",type=str,default=private_ids_json)
    # parser.add_argument("--rev-conv-id",type=str,default=conversation_id)
    # parser.add_argument("--base-dir", type=str, default=local)
    args = parser.parse_args()

    with open(args.params_json_path, "r") as f:
        params = json.load(f)

    with open(args.ids_json_path, "r") as f:
        ids = json.load(f)

    if args.log:
        if not os.path.exists(args.log_file):
            prev_req = set()
            file = open(args.log_file, "w")
            file.close()
        else:
            with open(args.log_file, "r") as last_log:
                lines = last_log.readlines()
                prev_req = set()
                for line in lines:
                    if not line.startswith("No"):
                        continue
                    prev_req.add(line[line.find(":") + 2 :].strip())

    # print(len(prev_req)+1)

    chat = ChatLoop(
        output_path=args.output_file,
        log_path=args.log_file,
        error_path=args.error_file,
        conversation_id=[None, None],
        access_token=ids["access_tokens"],
        begin_cnt=len(prev_req) + 1,
    )

    gt = params["guest_type"]
    req_opt = params["tour_places"]
    prompts = Prompt(
        main_prompt=params[args.main_prompt_name],
        guest_type=gt,
        total_words_limit=params["total_words_limit"],
        per_response_limit=params["per_response_limit"],
        specialty=params["specialty"],
        req_options=req_opt,
    )
    print(f'prompt: {params[args.main_prompt_name]}')
    print("----running----")
    try:
        if args.revgpt:
            asyncio.run(chat.start_async_chat(prompt_object=prompts, logs=prev_req))
        elif args.openai:
            chat.start_openai_chat(prompt_object=prompts, logs=prev_req)
    except KeyboardInterrupt as e:
        if chat.conversation_id[chat.chatbot_num] is not None:
            asyncio.run(chat.delete_conv())
