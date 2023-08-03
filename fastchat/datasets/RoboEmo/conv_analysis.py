import json, argparse

def make_cnt_dict():
    return {"right_cnt":0,"left_cnt":0,"tie_cnt":0,"both_bad_cnt":0}

def getStats(convs):
    res = {}
    models = None
    for line in convs:
        # every line in this file is a json
        cur_json = json.loads(line)
        if cur_json["session_id"] not in res.keys():
            res[cur_json["session_id"]] = make_cnt_dict()
        if models is None and "states" in cur_json.keys():
            models = [cur_json["states"][0]["model_name"],cur_json["states"][1]["model_name"]]
        if cur_json["type"] == "leftvote":
            res[cur_json["session_id"]]["left_cnt"] += 1
        elif cur_json["type"] == "rightvote":
            res[cur_json["session_id"]]["right_cnt"] += 1
        elif cur_json["type"] == "tievote":
            res[cur_json["session_id"]]["tie_cnt"] += 1
        elif cur_json["type"] == "bothbad_vote":
            res[cur_json["session_id"]]["both_bad_cnt"] += 1
    return res,models
    print(f'{models[0]} vote:{left_cnt}, {models[1]} vote:{right_cnt}, tievote:{tie_cnt}, bothbad_vote:{both_bad_cnt}')
    print(f'{models[0]} percentage:{((left_cnt+tie_cnt)/sum_cnt)*100}%, {models[1]} percentage:{((right_cnt+tie_cnt)/sum_cnt)*100}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json",type=str,default=None)
    args = parser.parse_args()

    assert args.input_json is not None, "Please input a json file path with option --input-json"

    assert args.input_json.endswith(".json")

    with open(args.input_json,'r') as r:
        convs = r.readlines()
        res,models = getStats(convs)
        # print(res)
        print(f'left model:{models[0]},right model:{models[1]}')
        for index,item in enumerate(res):
            temp = res[item]
            summ = sum(temp.values())
            print(f'No {index+1}: left_cnt={temp["left_cnt"]}\
                    right_cnt={temp["right_cnt"]} \
                    tie_cnt={temp["tie_cnt"]} \
                    both_bad_cnt={temp["both_bad_cnt"]} ')
            print(f'{models[0]} percentage:{((temp["left_cnt"]+temp["tie_cnt"])/summ)*100}%, {models[1]} percentage:{((temp["right_cnt"]+temp["tie_cnt"])/summ)*100}%')