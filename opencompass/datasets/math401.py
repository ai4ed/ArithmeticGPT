from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)

from .base import BaseDataset
import json
import re



@LOAD_DATASET.register_module()
class Math401Dataset(BaseDataset):

    @staticmethod
    def load(path: str):

        dataset = DatasetDict()
        with open(path, 'r') as json_file:
            data = json.load(json_file)
        raw_data = []
        for item in data:
            answer = item['response'].replace(',', '')
            answer = round(float(answer), 2)
            raw_data.append({
                'problem':
                item['query'],
                'answer':
                str(answer)
            })           
        dataset['test'] = Dataset.from_list(raw_data)
        dataset['train'] = Dataset.from_list(raw_data)
        return dataset


