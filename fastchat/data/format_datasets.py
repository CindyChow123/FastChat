import json
import argparse
from tqdm import tqdm

from datasets import load_dataset

dir = "../datasets/"

def JsonSingleData(data):
    js = json.dumps(data,indent=4,separators=(',',':'))
    return js

def load_data(name):
    return load_dataset(name)

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

def main():
    dataset = load_data(args.dataset_name)
    if (args.dataset_name == "conv_ai_2"):
        output = format_ConvAi2(dataset['train'])
        json.dump(output,open("train_"+args.out_file,"w"),indent=2)
    elif(args.dataset_name == "allenai/prosocial-dialog"):
        for key in dataset.shape.keys():
            output = format_prosocial_dialog(dataset[key])
            json.dump(output,open(dir+key+"_"+args.out_file,"w"),indent=2)
            print(f'{key}:{len(output)}')

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", type=str, default="allenai/prosocial-dialog")
    parser.add_argument("--out-file", type=str,default="prosocial_dialog_clean.json")
    args = parser.parse_args()
    main()

    
    