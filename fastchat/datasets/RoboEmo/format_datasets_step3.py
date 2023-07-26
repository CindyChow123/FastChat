"""
Step3: 
1.  To check dataset for conversation rounds length and roles
"""
from format_datasets import revchatgpt2json, add_arguments
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    args.json_file = "/data/Im-sys/FastChat/fastchat/datasets/RoboEmo/Host/Host_rev_chatGPT_davinci_all_25111.json"

    rvjson = revchatgpt2json(
        dataset_name=args.dataset_name,
        out_file=args.out_file,
        json_file=args.json_file,
        begin_cnt=args.begin_cnt,
        part=args.part
    )
    print(rvjson.check_dataset())