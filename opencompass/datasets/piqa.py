from datasets import load_dataset,Dataset
import json
from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class piqaDataset_V2(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            assert isinstance(example['label'], int)
            if example['label'] < 0:
                example['answer'] = 'NULL'
            else:
                example['answer'] = 'AB'[example['label']]
            example.pop('label')
            return example

        dataset = dataset.map(preprocess)
        return dataset


@LOAD_DATASET.register_module()
class piqaDataset_V3(BaseDataset):

    @staticmethod
    def load(path):
        labels = open(path + "/valid-labels.lst").readlines()
        dataset = []
        with open(path + "/valid.jsonl", 'r') as f:
            for idx, line in enumerate(f):
                data = json.loads(line)
                dataset.append({
                    'goal': data['goal'][0].upper() + data['goal'][1:],
                    'sol1': data["sol1"][0].upper() + data["sol1"][1:],
                    "sol2": data["sol2"][0].upper() + data["sol2"][1:],
                    "label": int(labels[idx])
                })
        dataset = Dataset.from_list(dataset)
        return dataset
    
@LOAD_DATASET.register_module()
class piqaDataset_V4(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            example['goal'] = example['goal'][0].upper() + example['goal'][1:]
            if example['goal'].endswith('?') or example['goal'].endswith('.'):
                example['sol1'] = example['sol1'][0].upper(
                ) + example['sol1'][1:]
                example['sol2'] = example['sol2'][0].upper(
                ) + example['sol2'][1:]
            else:
                example['sol1'] = example['sol1'][0].lower(
                ) + example['sol1'][1:]
                example['sol2'] = example['sol2'][0].lower(
                ) + example['sol2'][1:]
            return example

        dataset = dataset.map(preprocess)
        return dataset
