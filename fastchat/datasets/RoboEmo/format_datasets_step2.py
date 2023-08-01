"""
Step2: 
1.  To reformat the combined data text file into a json file that vicuna can accept as inputs
"""
from format_datasets import revchatgpt2json, add_arguments
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    args.dataset_name = "/data/Im-sys/FastChat/fastchat/datasets/RoboEmo/Host/Host_rev_chatGPT_davinci_all_noNo"
    args.out_file = "/data/Im-sys/FastChat/fastchat/datasets/RoboEmo/Host/Host_rev_chatGPT_davinci_all"

    rvjson = revchatgpt2json(
        dataset_name=args.dataset_name,
        out_file=args.out_file,
        json_file=args.json_file,
        begin_cnt=args.begin_cnt,
        part=args.part,
    )
    rvjson.make_vicuna_revChatGPT_dataset()
