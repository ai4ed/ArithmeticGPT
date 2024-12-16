import os.path as osp
import json
from datasets import Dataset
from datasets import DatasetDict, load_dataset

from opencompass.registry import LOAD_DATASET, TEXT_POSTPROCESSORS

from .base import BaseDataset
import re
from opencompass.utils.text_postprocessors import math_answer_extract,last_answer_postprocess


@LOAD_DATASET.register_module()
class MawpsDataset(BaseDataset):

    @staticmethod
    def load(path: str):
        raw_data = []
        # for split in ['validset', 'testset','trainset']:
        for split in ['testset']:
            filename = osp.join(path, f'{split}.json')
            with open(filename, encoding='utf-8') as f:
                reader = json.load(f)
                raw_data.extend(reader)
        dataset = Dataset.from_list(raw_data)
        return dataset

@TEXT_POSTPROCESSORS.register_module('mawps_dataset')
def first_option_postprocess(text: str) -> str:
    """Find first valid option for text."""

    patterns = [
        # r"####\s?(\-?\d+[\.|\,]?\d*)\s?",
        r"answer is (\-?\d+[\.|\,]?\d*)",
        r"answer is .*\=\s?(\-?\d+[\.|\,]?\d*)\$",
        r'\=\s?(\-?\d+[\.|\,]?\d*).*\n',
        r"(\-?\d+[\.|\,]?\d*)",
    ]

    regexes = [re.compile(pattern) for pattern in patterns]
    for regex in regexes:
        if regex == re.compile(r'\=\s?(\-?\d+[\.|\,]?\d*).*\n'):
            match_list = regex.findall(text)
            if match_list:
                outputs = match_list[-1]
                if '.' not in outputs:
                    outputs = outputs+'.0'
                return outputs     
        else:
            match = regex.search(text)
            if match:
                outputs = match.group(1).replace(',', '')
                outputs = outputs.strip('.0')
                if '.' not in outputs:
                    outputs = outputs+'.0'
                return outputs
    return ''


def mawps_postprocess(text: str) -> str:
    text = last_answer_postprocess(text)
    try:
        text = round(float(text), 2)
        if '.' not in str(text):
            text = str(text)+'.0'
        return text
    except ValueError:
        return text
