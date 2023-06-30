import json
from dataclasses import field,dataclass
import argparse
from tqdm import tqdm
from typing import List, Tuple, Any, Dict, Set, Optional

# from datasets import load_dataset


def JsonSingleData(data):
    js = json.dumps(data,indent=4,separators=(',',':'))
    return js

def format_ConvAi2(dataset):
    output = []
    with tqdm(total = dataset.num_rows) as pbar:
        for data in dataset:
            dataclean = dict()
            dataclean['id'] = data['dialog_id'] 
            dataclean['conversations'] = []
            for item in data['dialog']:
                cleanitem = dict()
                cleanitem['from'] = item['sender']
                cleanitem['value'] = item['text']
                dataclean['conversations'].append(cleanitem)
            output.append(dataclean)
        pbar.update(1)
    return output

def format_prosocial_dialog(dataset):
    output = []
    id = 0
    with tqdm(total = dataset.num_rows) as pbar:
        dataclean = dict()
        dataclean['id'] = "identity_"+str(id)
        dataclean['conversations'] = []
        for data in dataset:
            if(data['dialogue_id'] != id):
                output.append(dataclean)
                id += 1
                dataclean = dict()
                dataclean['id'] = "identity_"+str(id)
                dataclean['conversations'] = []

            dataclean['conversations'].append({"from":"human","value":data["context"]})
            dataclean['conversations'].append({"from":"gpt","value":data["response"]})

            pbar.update(1)
            
    return output

def main_json_dataset():
    dataset = load_dataset(args.dataset_name)
    if (args.dataset_name == "conv_ai_2"):
        output = format_ConvAi2(dataset['train'])
        json.dump(output,open("train_"+args.out_file,"w"),indent=2)
    elif(args.dataset_name == "allenai/prosocial-dialog"):
        for key in dataset.shape.keys():
            output = format_prosocial_dialog(dataset[key])
            json.dump(output,open(dir+key+"_"+args.out_file,"w"),indent=2)
            print(f'{key}:{len(output)}')

@dataclass
class revchatgpt2json:
    roles: Dict= field(default_factory=lambda:{"human":0,"gpt":1})
    dir: str="../datasets/"
    users: List[str] = field(default_factory=lambda:["Guest","Old man","Old Man","President"])
    bots: List[str] = field(default_factory=lambda:["Bot","Tour Guide","Tour guide","Guide"])

    def isUser(self,line:str):
        '''Check whether the speaking role is USER'''
        for user in self.users:
            if line.startswith(user):
                return True
        return False

    def isBot(self,line:str):
        '''Check whether the speaking role is GPT'''
        for bot in self.bots:
            if line.startswith(bot):
                return True
        return False

    def add_line(self,conv:dict,line:str, last_role:int,from_who:str):
        '''Add a line to the json object with correct speaking role'''
        line = line[line.find(':')+1:].strip()
        if line[-3:].upper() == "END": line = line[:-3]
        if last_role==self.roles[from_who] and len(conv["conversations"])!=0:
            conv["conversations"][-1]["value"] += line
        else:
            conv["conversations"].append({"from":from_who,"value":line})
        return conv

    def main_revChatGPT_dataset(self):
        '''Format the string dataset into a valid json file'''
        with open(self.dir+args.dataset_name,'r') as fread:
            contents = fread.readlines()
            json_content = []
            conv = {}
            id = args.begin_cnt
            total_dia = 0
            last_role = -1
            for line in contents:
                if(line.startswith("BEGIN") or line.startswith("EGIN")):
                    if(len(conv)>0 and len(conv["conversations"])>1):
                        json_content.append(conv)
                        id += 1
                        total_dia += 1
                    conv = {}
                    conv["id"]="identity_"+str(id)
                    conv["conversations"] = []
                elif self.isUser(line):
                    conv = self.add_line(conv,line,last_role,from_who="human")
                    last_role=0
                elif self.isBot(line):
                    conv = self.add_line(conv,line,last_role,from_who="gpt")
                    last_role = 1
                    
            json.dump(json_content,open(self.dir+args.out_file+f'{total_dia}{"_part" if args.part else ""}.json',"w"),indent=2)
            print(f'Finish {total_dia}.')

    def check_dataset(self):
        '''Check dataset's validity'''
        dataset = json.load(open(self.dir+args.json_file,'r'))
        too_short = self.check_too_short(dataset)
        role_check = self.check_role_round(dataset)
        if too_short and role_check:
            return True
        else:
            return False
        
    def check_too_short(self,dataset):
        '''Check whether conversations are all at least two rounds'''
        too_short = []
        for d in dataset:
            conv = d["conversations"]
            id = d["id"]
            if len(conv) <= 1:
                too_short.append(id)
                continue
        if len(too_short) == 0: return True
        else: 
            print(f'Too short: {too_short}')
            return False
        
    def check_role_round(self,dataset):
        '''Check whether it's two roles talking interchangeably'''
        wrong_role=[]
        for d in dataset:
            conv = d["conversations"]
            id = d["id"]
            last_role = 1
            for i,c in enumerate(conv):
                cur_role = self.roles[c["from"]]
                if cur_role+last_role != 1:
                    wrong_role.append(id)
                last_role = cur_role
        if len(wrong_role) == 0:
            return True
        else:
            print(f'wrong role: {wrong_role}')
            return False
            
    def combine_files(self,filenames):
        '''Concat dataset files, preprocess the in-line BEGIN and END'''
        with open(self.dir+args.out_file,"w") as f:
            for file in filenames:
                lines = open(file,"r").readlines()
                for line in lines:
                    begin_pos = line.find("BEGIN")
                    end_pos = line.find("END")
                    # if BEGIN is in the line content, seperate two lines
                    if begin_pos != -1 and begin_pos != 0:
                        f.write(line[:begin_pos]+'\n')
                        f.write("BEGIN\n")
                        f.write(line[begin_pos+1:])
                    # if END is in the line content, remove it
                    elif end_pos != -1 and end_pos != 0:
                        line.replace("END","r")
                        f.write(line)
                    else:
                        f.write(line)

    def show_all_roles(self,filename):
        '''Collect all the conversation roles used for future json parsing'''
        with open(self.dir+filename,'r') as fread:
            contents = fread.readlines()
            heads = []
            for content in contents:
                head = content[0:content.find(':')]
                if head.find("No")==-1 and head.find("BEGIN")==-1 and head.find("END")==-1 and head not in heads:
                    heads.append(head)
        for h in heads:
            print(h)
                    
    def delete_before(self, num:int):
        with open(self.dir+args.out_file+"concat","w") as wf:
            with open(self.dir+args.dataset_name,'r') as f:
                write = False
                for line in f.readlines():
                    if write:
                        wf.write(line)
                    if line.startswith("No"):
                        if line == f'No {num}\n':
                            print(line)
                            write = True
                            wf.write(line)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="RoboEmo/Host_concat")
    parser.add_argument("--out-file", type=str,default="RoboEmo/Host_")
    parser.add_argument("--json-file",type=str,default="RoboEmo/Host_13357.json")
    parser.add_argument("--begin-cnt", type=int, default=0)
    parser.add_argument("--part",type=bool, default=False)
    args = parser.parse_args()
    
    rvjson = revchatgpt2json()

    '''Step one, gather all data, check the role's name and edit users[], bots[]'''
    # files = ["/data/Im-sys/FastChat/fastchat/datasets/RoboEmo/Host",
    #          "/data/Im-sys/FastChat/fastchat/datasets/RoboEmo/Host_rev_add_davinci",
    #          "/data/Im-sys/FastChat/fastchat/datasets/RoboEmo/Host_rev_add_davinci_0625",
    #          "/data/Im-sys/FastChat/fastchat/datasets/RoboEmo/Host_rev_add_davinci_0629"]
    # rvjson.combine_files(files)
    # rvjson.show_all_roles(args.out_file)
    '''Extra, if want to delete the first few roles'''
    # rvjson.delete_before(11638)

    '''Step two, format into json file'''
    rvjson.main_revChatGPT_dataset()
    
    '''Check dataset'''
    # print(rvjson.check_dataset())
