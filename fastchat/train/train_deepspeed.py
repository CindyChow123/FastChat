# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import copy
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence
import random
import math
import os
import time
import sys

# Must before fastchat importing
sys.path.append("/TTS_personal_jiahui.ni/Im-sys/FastChat/")


import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import transformers
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

from transformers.deepspeed import HfDeepSpeedConfig
import deepspeed

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
date = time.strftime("%m%d%H%M")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    ds_config_path: str = field(
        default=None, metadata={"help": "Path of deepspeed config json."}
    )


def set_random_seed(seed):
    if seed is not None:
        transformers.set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


local_rank = None
training_args = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def save_hf_format(model, tokenizer, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, "module") else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(training_args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)
    torch.save(training_args, os.path.join(output_dir, "training_args.bin"))


def get_optimizer_grouped_parameters(
    model, weight_decay, no_decay_name_list=["bias", "LayerNorm.weight"]
):
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("vicuna")
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

            cur_len += round_len
        target[cur_len:] = IGNORE_TOKEN_ID

        if False:
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")
    raw_data = json.load(open(data_args.data_path, "r"))

    # Split train/test
    perm = np.random.permutation(len(raw_data))
    split = int(len(perm) * 0.98)
    train_indices = perm[:split]
    eval_indices = perm[split:]
    train_raw_data = [raw_data[i] for i in train_indices]
    eval_raw_data = [raw_data[i] for i in eval_indices]
    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")

    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def eval(model, eval_dataloader, epoch):
    model.eval()
    losses = 0
    rank0_print(
        f"***** Evaluating Epoch {epoch+1}/{training_args.num_train_epochs} *****",
        model.local_rank,
    )
    for step, batch in enumerate(eval_dataloader):
        batch = batch.to(model.local_rank)
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses += loss.float()
    losses = losses / step + 1
    rank0_print(f"Eval losses: {losses}")


def train(model, tokenizer, train_dataloader, eval_dataloader):
    eval(model=model, eval_dataloader=eval_dataloader, epoch=0)
    for epoch in range(int(training_args.num_train_epochs)):
        rank0_print(
            f"Beginning of Epoch {epoch+1}/{training_args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            model.local_rank,
        )
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = batch.to(model.local_rank)
            outputs = model(**batch)
            loss = outputs.loss
            model.backward(loss)
            model.step()

            # save and log every __ steps
            if step % training_args.eval_steps == 0:
                eval(model=model, eval_dataloader=eval_dataloader, epoch=epoch)
            if step % training_args.save_steps == 0:
                save_hf_format(
                    model=model, tokenizer=tokenizer, sub_folder=date + str(step)
                )
    return model


def main():
    global local_rank
    global training_args
    print("*****Reading arguments*****")
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    ds_config = (
        training_args.ds_config_path
    )  # deepspeed config object or path to the file
    training_args.deepspeed = ds_config
    set_random_seed(training_args.seed)
    local_rank = training_args.local_rank
    # keep this one alive?
    dschf = HfDeepSpeedConfig(ds_config)

    print("*****Loading model*****")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False

    print("*****Loading tokenizer*****")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    data_collator = transformers.DataCollatorWithPadding(
        tokenizer=tokenizer, max_length=training_args.model_max_length
    )

    print("*****Processing dataset*****")
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    train_dataloader = DataLoader(
        data_module["train_dataset"],
        collate_fn=data_collator,
        sampler=DistributedSampler(data_module["train_dataset"]),
        batch_size=training_args.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(
        data_module["eval_dataset"],
        collate_fn=data_collator,
        sampler=DistributedSampler(data_module["eval_dataset"]),
        batch_size=training_args.per_device_eval_batch_size,
    )

    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, training_args.weight_decay
    )

    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=training_args.learning_rate
    )

    num_update_steps_per_epoch = math.ceil(
        len(data_module["train_dataset"]) / training_args.gradient_accumulation_steps
    )

    lr_scheduler = transformers.get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=training_args.warmup_ratio * num_update_steps_per_epoch,  # ???
        num_training_steps=training_args.num_train_epochs * num_update_steps_per_epoch,
    )

    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=training_args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
    )

    local_rank = model_engine.local_rank

    if training_args.gradient_checkpointing:
        model_engine.gradient_checkpointing_enable()

    rank0_print("***** Running training *****", local_rank)

    final_model = train(
        model=model_engine,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )

    if training_args.output_dir is not None:
        rank0_print("Saving the final model", final_model.local_rank)
        if final_model.local_rank == 0:
            save_hf_format(
                final_model,
                tokenizer=tokenizer,
                args=training_args,
                sub_folder=date + "last",
            )


if __name__ == "__main__":
    main()
