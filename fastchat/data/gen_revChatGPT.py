import asyncio, dataclasses,requests, openai, json, time,argparse,os,httpx
from typing import List, Tuple, Any, Dict, Set

from revChatGPT.V1 import Chatbot, AsyncChatbot
'''
python gen_revChatGPT.py >> host_0614_1136.log 2>&1'''


access_tokens = []
conversation_ids =[]
psy_input = "Please generate a converstaion between a considerate expressive therapist and a patient. Keep each sentence to 40 words and total conversation to 150 words. Conversations should be related to "
psy_topic = [
    "alcohol addiction",
    "anxiety disorder",
    "bereavement",
    "body dysmorphic disorder",
    "depression",
    "Insomnia",
    "Hypersomnia",
    "Idiopathic hypersomnia",
    "Kleine–Levin syndrome",
    "Insufficient sleep syndrome",
    "Narcolepsy",
    "Restless legs syndrome",
    "Sleep apnea",
    "Night terrors (sleep terrors)",
    "Exploding head syndrome",
    "panic disorders",
    "obsessive compulsive disorder",
    "post-traumatic stress disorder",
    "Bipolar affective disorder",
    "Paranoia",
    "Hypochondriasis",
    "Somatization disorder",
    "Conversion disorder (Functional Neurological Symptom Disorder)",
    "Factitious disorder imposed on self (munchausen syndrome)",
    "Factitious disorder imposed on another (munchausen by proxy)",
    "Pain disorder",
    "Pica (disorder)",
    "Rumination syndrome",
    "Avoidant/restrictive food intake disorder",
    "Anorexia nervosa",
    "Binge eating disorder",
    "Bulimia nervosa",
    "Purging disorder,"
    "Diabulimia",
    "Night eating syndrome",
    "Orthorexia nervosa",
    "study stress",
    "relationship betrayal",
    "emotional break-up",
    "fears to talk to other people",
    "marriage and divorce",
    "domestic violence"
]
host_input = [
  '''Please act as an enthusiastic, considerate, and professional guide full of emotion and energy. I will ask you to generate conversations between you and the guest in different locations. 
      Guest type: lady with a baby
      Total words limit: 300 words
      Words limit per response: 60 words
      Special Requirements: Try best to use different greetings, adjectives and adverbs and goodbyes in each conversation
      Who to begin conversation: Guest
      Who to end conversation: Guide
      My first request: the Griffith Park'''
]
recp_input=[
  '''Please act as an enthusiastic receptionist full of emotion and energy. I will ask you to generate conversations between you and the guest in different locations. 
      Guest type: lady with a baby
      Total words limit: 300 words
      Words limit per response: 60 words
      Special Requirements: Try best to use different greetings, adjectives and adverbs and goodbyes in each conversation
      My first request: in a hotel'''
]
roles = [
  ["Tourist","Hostess"],
  ["Guest","Receptionist"]
]

recp_main_prompt = "Please act as an enthusiastic receptionist full of emotion and energy. I will ask you to generate conversations between you (bot) and the guest (guest) in different locations. You can add whatever details."
host_main_prompt = "Please act as an enthusiastic tour guide full of emotion and energy. I will ask you to generate conversations between you (bot) and the tourists (guest) in different locations. Please introduce the place in detail."

# openai.api_key = ""
openai.api_key = 


@dataclasses.dataclass
class Prompt():
  '''A class for generating promp pattern'''
  main_prompt: str
  guest_type: List[str]
  total_words_limit: List[int]
  per_response_limit: List[int]
  specialty: str
  req_options:List[str]
  gt_init: int=0
  total_init: int=0
  per_init: int=0
  opt_init: int=0

  type_pre: str="\nGuest type: "
  total_pre: str="\nTotal words limit: "
  per_pre: str="\nMinimum words per response: "
  special_pre: str="\nSpecial requirements: "
  # who_to_ask: str="\nWho to ask: Guest"
  # who_to_answer: str="\nWho to answer: Robo"
  req_pre: str="\nCurrent request: in the "
  
  def get_prompts(self):

    strr = self.main_prompt
    i=self.gt_init
    while(i<len(self.guest_type)):
      j=self.total_init
      while(j<len(self.total_words_limit)):
        k=self.per_init
        while(k<len(self.per_response_limit)):
          m=self.opt_init
          while(m<len(self.req_options)):
            guest = self.type_pre + self.guest_type[i]
            total = self.total_pre+str(self.total_words_limit[j])
            per = self.per_pre+str(self.per_response_limit[k])
            special = self.special_pre+self.specialty
            req = self.req_pre+self.req_options[m]
            yield strr+guest+total+per+special+req, f'gt-{i}-total-{j}-per-{k}-opt-{m}'
            m += 1
          k+=1
        j+=1
      i+=1
          

@dataclasses.dataclass
class ChatLoop():
  '''A class for chatgpt chatting'''
  output_path: str
  log_path: str
  conversation_id: List[str]
  access_token: List[str] # ?? using List + field?
  begin_cnt: int=1
  openai_model: str="gpt-3.5-turbo-16k"

  async def start_async_chat(self,prompt_object:Prompt, logs:Set[str]):
    chatbot_gmail = AsyncChatbot(config = {
      "access_token": self.access_token[0],
      "conversation_id": self.conversation_id[0]
    })
    chatbot_usc = AsyncChatbot(config = {
      "access_token": self.access_token[1],
      "conversation_id": self.conversation_id[1]
    })
    chatbots = [chatbot_gmail,chatbot_usc]

    t=0
    with open(self.log_path,'a') as log:
      with open(self.output_path,'a') as f:
        cnt = self.begin_cnt
        chatbot_num = 1
        # times = 10
        for prompt, extra in prompt_object.get_prompts():
          if extra not in logs:
            prev_text=""

            try:
              print(chatbot_num)
              # times -= 1
              # if times == 0: break
              # raise httpx.HTTPStatusError(response)
              async for data in chatbots[chatbot_num].ask(prompt):
                if self.conversation_id[chatbot_num] is None:
                    self.conversation_id[chatbot_num] = data["conversation_id"]
                    print(f'Conv id {self.conversation_id[chatbot_num]}')
                message = data["message"][len(prev_text) :]
                
                if message.endswith("END"):
                  print(message, end="\n", flush=True,file=f)
                else:
                  print(message, end="", flush=True,file=f)

                prev_text = data["message"]
              print(f'No.{cnt}: {extra}',flush=True,file=log)
              cnt += 1
              time.sleep(20)

              if cnt-self.begin_cnt >= 80: chatbot_num = 1-chatbot_num
            except Exception as exc:
              print(f'Error while requesting {exc}')
              chatbot_num = 1-chatbot_num
              async for data in chatbots[chatbot_num].ask(prompt):
                if self.conversation_id[chatbot_num] is None:
                    self.conversation_id[chatbot_num] = data["conversation_id"]
                    print(f'Conv id {self.conversation_id[chatbot_num]}')
                message = data["message"][len(prev_text) :]
                
                if message.endswith("END"):
                  print(message, end="\n", flush=True,file=f)
                else:
                  print(message, end="", flush=True,file=f)

                prev_text = data["message"]
              print(f'No.{cnt}: {extra}',flush=True,file=log)
              cnt += 1
              time.sleep(20)
              continue
    
  def start_openai_chat(self,prompt_object:Prompt,logs:Set[str]):
    with open(self.output_path,'a') as f:
      cnt = self.begin_cnt
      for prompt, extra in prompt_object.get_prompts():
        if extra not in logs:
          print(f'No.{cnt}: {extra}',flush=True)
          # print("Conversation:",file=f)
          # create a chat completion
          chat_completion = openai.ChatCompletion.create(model=self.openai_model, messages=[{"role": "user", "content": prompt}])
          # # print the chat completion
          print(chat_completion.choices[0].message.content,flush=True,file=f)
          cnt += 1
          time.sleep(20)



if __name__=="__main__":

  openai_output = "/data/Im-sys/FastChat/fastchat/datasets/RoboEmo/Host"
  openai_log = "/data/Im-sys/FastChat/fastchat/data/host_0614_1136_copy.log"
  rev_output = "/data/Im-sys/FastChat/fastchat/datasets/RoboEmo/Host_rev_copy"
  rev_log = "/data/Im-sys/FastChat/fastchat/data/host_0619_rev.log"
  rcpt_place_json = "/data/Im-sys/FastChat/fastchat/datasets/RoboEmo/Rcpt_places.json"

  parser = argparse.ArgumentParser()
  parser.add_argument("--output-file",type=str,default=rev_output)
  parser.add_argument("--log",action="store_true",default=False)
  parser.add_argument("--log-file",type=str,default=openai_log)
  parser.add_argument("--openai",action="store_true",default=False)
  parser.add_argument("--revgpt",action="store_true",default=False)
  # parser.add_argument("--rev-conv-id",type=str,default=conversation_id)
  args = parser.parse_args()

  with open(rcpt_place_json,'r') as f:
    params = json.load(f)

  if args.log:
    if not os.path.exists(args.log_file):
      prev_req = set()
      file = open(args.log_file, 'w')
      file.close()
    else:
      with open(args.log_file,'r') as last_log:
        lines = last_log.readlines()
        prev_req = set()
        for line in lines:
          if not line.startswith("No"): continue
          prev_req.add(line[line.find(':')+2:].strip())    
    
  
  # print(len(prev_req)+1)

  chat = ChatLoop(output_path=args.output_file,
                  log_path = args.log_file,
                  conversation_id=conversation_ids,
                  access_token=access_tokens,
                  begin_cnt=len(prev_req)+1)

  gt = params['guest_type']
  req_opt = params['tour_places']
  prompts = Prompt(
    main_prompt= host_main_prompt,
    guest_type=gt,
    total_words_limit=[100,200,300,400,500],
    per_response_limit=[5,15,25,35,45],
    specialty="Try best to use different greetings, adjectives and adverbs and goodbyes in each conversation. \
      Begin the conversation by the \"guest\" greeting the \"bot\", end the conversation by the \"guest\" bidding farewell at the \"bot\" with the \"bot\" answering back.\
      Begin each conversation by typing \"BEGIN\n\" in a separate line, end each conversation by typing \"END\n\" in a separate line. Only outputs the dialogue sentences.",
    req_options=req_opt,
  )

  if args.revgpt:      
    asyncio.run(chat.start_async_chat(prompt_object=prompts,logs=prev_req))
  elif args.openai:
    chat.start_openai_chat(prompt_object=prompts,logs=prev_req)
  
  
