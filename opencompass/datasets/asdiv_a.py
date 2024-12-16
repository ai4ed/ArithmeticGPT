import os
import json
from datasets import Dataset
from datasets import DatasetDict, load_dataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import  (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)
from opencompass.utils.text_postprocessors import math_answer_extract,last_answer_postprocess


from .base import BaseDataset
import re

@LOAD_DATASET.register_module()
class AsdivDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        raw_data = []
        file_path = os.path.join(path, f'{name}.jsonl')
        with open(file_path, 'r',encoding='utf-8') as fr:
            reader = fr.readlines()
            new_reader = []
            for sample in reader:
                sample_dict = json.loads(sample)
                new_sample = {}
                new_sample['id'] = sample_dict ['id']
                new_sample['problem'] = sample_dict ['problem']
                new_sample['answer'] = str(sample_dict ['answer'])
                new_reader.append(new_sample)
            raw_data.extend(new_reader)
        dataset = Dataset.from_list(raw_data)
        return dataset

@TEXT_POSTPROCESSORS.register_module()
def asdiv_first_option_postprocess(text: str) -> str:
    """Find first valid option for text."""

    patterns = [
        # r"####\s?(\-?\d+[\.|\,]?\d*)\s?",
        r"answer is (\-?\d+[\.|\,]?\d*)",
        r"=> (\-?\d+[\.|\,]?\d*)",
        r'=\s*(\-?\d+[\.|\,]?\d*)\n',
        r"(\-?\d+[\.|\,]?\d*)\n",
        r"(\-?\d+[\.|\,]?\d*)",
    ]

    regexes = [re.compile(pattern) for pattern in patterns]
    for regex in regexes:
        match = regex.search(text)
        if match:
            outputs = match.group(1)
            outputs = outputs.replace(',', '').strip().strip('\n')
            if outputs.endswith('.'):
                outputs = outputs.strip('.')
            return outputs
    return ''

def asdiv_postprocess(text: str) -> str:
    text = last_answer_postprocess(text)
    try:
        if '.' in str(text):
            text = round(float(text), 2)
            return str(text)
        return str(text)
    except ValueError:
        return str(text)