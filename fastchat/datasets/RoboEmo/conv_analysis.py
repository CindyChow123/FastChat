import json, argparse

def getStats(convs):
    right_cnt = 0
    left_cnt = 0
    tie_cnt = 0
    both_bad_cnt = 0
    models = None
    for line in convs:
        # every line in this file is a json
        cur_json = json.loads(line)
        if models is None and "models" in cur_json.keys():
            models = cur_json["models"]
        if cur_json["type"] == "leftvote":
            left_cnt += 1
        elif cur_json["type"] == "rightvote":
            right_cnt += 1
        elif cur_json["type"] == "tievote":
            tie_cnt += 1
        elif cur_json["type"] == "bothbad_vote":
            both_bad_cnt += 1
    sum_cnt = right_cnt+left_cnt+tie_cnt+both_bad_cnt
    print(f'rightvote:{right_cnt}, leftvote:{left_cnt}, tievote:{tie_cnt}, bothbad_vote:{both_bad_cnt}')
    print(f'{models[0]} percentage:{((left_cnt+tie_cnt)/sum_cnt)*100}%, {models[1]} percentage:{((right_cnt+tie_cnt)/sum_cnt)*100}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json",type=str,default=None)
    args = parser.parse_args()

    assert args.input_json is not None, "Please input a json file path with option --input-json"

    assert args.input_json.endswith(".json")

    with open(args.input_json,'r') as r:
        convs = r.readlines()
        getStats(convs)