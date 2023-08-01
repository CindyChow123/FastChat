"""
Step1: 
1.  To combine all text files you want to use to generate the json file
2.  To see all the speaking roles ChatGPT has generated for conversations
"""
from format_datasets import revchatgpt2json, add_arguments
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()

    args.out_file = "/data/Im-sys/FastChat/fastchat/datasets/RoboEmo/Host/Host_rev_chatGPT_davinci_all_noNo"

    rvjson = revchatgpt2json(
        dataset_name=args.dataset_name,
        out_file=args.out_file,
        json_file=args.json_file,
        begin_cnt=args.begin_cnt,
        part=args.part,
    )

    files = [
        "/data/Im-sys/FastChat/fastchat/datasets/RoboEmo/Host/Host_rev_add_davinci_0_5765",
        "/data/Im-sys/FastChat/fastchat/datasets/RoboEmo/Host/Host_rev_chatGPT_davinci_5766_6312",
        "/data/Im-sys/FastChat/fastchat/datasets/RoboEmo/Host/Host_rev_chatGPT_davinci_6313_25200",
    ]
    rvjson.combine_files(files)
    rvjson.show_all_roles(args.out_file)
