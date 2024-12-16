#ref from https://huggingface.co/fireballoon/baichuan-vicuna-7b/blob/main/train_vicuna.py
# from fastchat.train.llama_flash_attn_monkey_patch import (
#     replace_llama_attn_with_flash_attn,
# )

# replace_llama_attn_with_flash_attn()


import copy
import math
from dataclasses import dataclass, field
import json
import pathlib
from typing import Dict, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


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


local_rank = None
dummy_message = [{"from": "human", "value": "Who are you?"},
                 {"from": "gpt", "value": "I am GPT, a language model trained by researchers"},
                 {"from": "human", "value": "What can you do?"},
                 {"from": "gpt", "value": "I can solve Math problems and chat with you."}]



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


def last_index(lst, value):
    return next((len(lst) - i - 1 for i, x in enumerate(lst[::-1]) if x != value), -1)


def safe_ids(ids, max_value, pad_id):
    return [i if i < max_value else pad_id for i in ids]


def tokenize(messages, tokenizer):
    roles = {"human": "USER", "gpt": "ASSISTANT"}
    input_ids = []
    labels = []
    system = "A chat between a curious user and an artificial intelligence assistant. " \
             "The assistant gives helpful, detailed, and polite answers to the user's questions."
    system = ''
    system_ids = tokenizer.encode(system, add_special_tokens=False)
    input_ids += system_ids
    labels += [IGNORE_TOKEN_ID] * len(system_ids)
    for i, turn in enumerate(messages):
        role = roles.get(turn['from'], 'USER')
        content = turn['value']
        content = content.strip()
        if role == 'ASSISTANT':
            role_id = [196]
        else:
            role_id = [195]
        # role_ids = tokenizer.encode(role + ":", add_special_tokens=False)
        content_ids = tokenizer.encode(content, add_special_tokens=False, truncation=True, max_length=tokenizer.model_max_length)
        input_ids += role_id + content_ids
        if role == 'ASSISTANT':
            labels += [IGNORE_TOKEN_ID] * len(role_id) + content_ids
        else:
            if i == 0:
                labels += [IGNORE_TOKEN_ID] * (len(role_id) + len(content_ids))
            else:
                labels += [tokenizer.eos_token_id] + [IGNORE_TOKEN_ID] * (len(content_ids))

    if tokenizer.add_bos_token:
        input_ids = [tokenizer.bos_token_id] + input_ids
        labels = [IGNORE_TOKEN_ID] + labels

    input_ids.append(tokenizer.eos_token_id)
    labels.append(tokenizer.eos_token_id)
    
    input_ids = input_ids[:tokenizer.model_max_length]
    labels = labels[:tokenizer.model_max_length]

    input_ids += [tokenizer.pad_token_id] * (tokenizer.model_max_length - len(input_ids))
    labels += [IGNORE_TOKEN_ID] * (tokenizer.model_max_length - len(labels))
        
    # rank0_print("*"*30)
    # rank0_print(labels)
    # rank0_print(input_ids)
    # rank0_print("-"*30)
    
    if len(labels) == 0:
        return tokenize(dummy_message, tokenizer)

    input_ids = torch.tensor([input_ids])
    labels = torch.tensor([labels])
    # torch tensor 
    attention_mask = input_ids.ne(tokenizer.pad_token_id)

    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask
    )
class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = tokenize(self.raw_data[i]["conversations"], self.tokenizer)
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

    data_sources = json.load(open(data_args.data_path, "r"))

    raw_data = []
    total_token_num = 0
    for data_source in data_sources:
        path = data_source["path"]
        sample_ratio = float(data_source["sample_ratio"])
        one_data = json.load(open(path, "r"))
        one_data = one_data[: int(len(one_data) * sample_ratio)]  # 顺序采样
        raw_data += one_data
        print(f"{path} has {len(one_data)} data, sample ratio {sample_ratio}")

    print("total data:", len(raw_data))

    # Split train/test
    # perm = np.random.permutation(len(raw_data))
    # split = int(len(perm) * 0.98)
    # train_indices = perm[:split]
    # eval_indices = perm[split:]
    train_raw_data = raw_data[:int(len(raw_data)*0.98)]
    eval_raw_data = raw_data[int(len(raw_data)*0.98):]
    # train_raw_data = [raw_data[i] for i in train_indices]
    # eval_raw_data = [raw_data[i] for i in eval_indices]
    rank0_print(f"#train {len(train_raw_data)}, #eval {len(eval_raw_data)}")

    train_dataset = dataset_cls(train_raw_data, tokenizer=tokenizer)
    eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    if "baichuan" in model_args.model_name_or_path or "Baichuan" in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            trust_remote_code=True,
            use_fast=False
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    if "Baichuan" in model_args.model_name_or_path:
        # baichuan has pad_token 
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True
    )

    special_tokens_dict = {
        "additional_special_tokens": [
            "<thought>",
            "</thought>",
            "<API>",
            "</API>",
            "<pad>"
        ]
    }
    
    tokenizer.add_special_tokens(special_tokens_dict)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    # old_embed = model.model.embed_tokens
    # new_embed = torch.nn.Embedding(len(tokenizer), 5120, device=old_embed.weight.device, dtype=old_embed.weight.dtype)
    # torch.nn.init.kaiming_uniform_(new_embed.weight, a=math.sqrt(5))
    # new_embed.weight.data[:old_embed.weight.size()[0], :] = old_embed.weight.data[:old_embed.weight.size()[0], :]
    # model.model.embed_tokens = new_embed
    
    # new_para = torch.nn.Parameter(torch.empty(len(tokenizer), 5120))
    # torch.nn.init.kaiming_uniform_(new_para, a=math.sqrt(5))
    # old_para = model.lm_head.weight
    # new_para.data[:old_para.size()[0], :] = old_para[:, :]
    # model.lm_head.weight = new_para
    
    # model.model.config.vocab_size = len(tokenizer)
    # model.config.vocab_size = len(tokenizer)
    # model.model.vocab_size = len(tokenizer)
    
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
