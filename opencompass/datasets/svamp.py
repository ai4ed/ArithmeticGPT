from datasets import Dataset, DatasetDict

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)

from .base import BaseDataset
import json
import re



@LOAD_DATASET.register_module()
class SVAMPDataset(BaseDataset):

    @staticmethod
    def load(path: str):

        dataset = DatasetDict()
        data =  []
        with open(path, 'r') as json_file:
            for line in json_file:
                dt = json.loads(line)
                data.append(dt)
        raw_data = []
        for item in data:
            # print(item)
            raw_data.append({
                'problem':
                item['problem'],
                'answer':
                item['answer']
            })           
        dataset['test'] = Dataset.from_list(raw_data)
        dataset['train'] = Dataset.from_list(raw_data)
        return dataset

