import json, argparse

def getStats(convs):
    right_cnt = 0
    left_cnt = 0
    tie_cnt = 0
    both_bad_cnt = 0
    for line in convs:
        if line.find("leftvote") != -1:
            left_cnt += 1
        if line.find("rightvote") != -1:
            right_cnt += 1
        if line.find("tievote") != -1:
            tie_cnt += 1
        if line.find("bothbad_vote") != -1:
            both_bad_cnt += 1
    sum_cnt = right_cnt+left_cnt+tie_cnt+both_bad_cnt
    print(f'rightvote:{right_cnt}, leftvote:{left_cnt}, tievote:{tie_cnt}, bothbad_vote:{both_bad_cnt}')
    print(f'right percentage:{((right_cnt+tie_cnt)/sum_cnt)*100}%, left percentage:{((left_cnt+tie_cnt)/sum_cnt)*100}%')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json",type=str,default=None)
    args = parser.parse_args()

    assert args.input_json is not None, "Please input a json file path with option --input-json"

    assert args.input_json.endswith(".json")

    with open(args.input_json,'r') as r:
        convs = r.readlines()
        getStats(convs)