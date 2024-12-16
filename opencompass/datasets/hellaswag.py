import json

from datasets import Dataset, load_dataset, DatasetDict
from opencompass.registry import LOAD_DATASET

from .base import BaseDataset


@LOAD_DATASET.register_module()
class hellaswagDataset(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            for i in range(4):
                example[chr(ord('A') + i)] = example['endings'][i]
            return example

        dataset = dataset.map(preprocess).remove_columns(['endings'])
        return dataset


@LOAD_DATASET.register_module()
class hellaswagDataset_V1(BaseDataset):

    @staticmethod
    def load(path: str):

        dataset = DatasetDict()

        datalist =  []
        datapath = path
        # print("datapath : ", datapath)
        with open(datapath, 'r') as json_file:
            for line in json_file:
                data = json.loads(line)
                datalist.append(data)

        raw_data = []
        for item in datalist:
            answer = item['gold']
            choice = ['A', 'B', 'C', 'D']
            label = choice[int(answer)]
            # print('answer-label: ', answer, '-', label)
            raw_data.append({
                'ctx': item['query'],
                "A": item["choices"][0],
                "B": item["choices"][1],
                "C": item["choices"][2],
                "D": item["choices"][3],
                'label': label
            })           
        dataset['test'] = Dataset.from_list(raw_data)
        dataset['train'] = Dataset.from_list(raw_data)
        return dataset

@LOAD_DATASET.register_module()
class hellaswagDataset_V2(BaseDataset):

    @staticmethod
    def load(**kwargs):
        dataset = load_dataset(**kwargs)

        def preprocess(example):
            for i in range(4):
                example[chr(ord('A') + i)] = example['endings'][i]
            if example['label']:
                example['label'] = 'ABCD'[int(example['label'])]
            else:
                example['label'] = 'NULL'
            return example

        dataset = dataset.map(preprocess).remove_columns(['endings'])
        return dataset


@LOAD_DATASET.register_module()
class hellaswagDataset_V3(BaseDataset):

    @staticmethod
    def load(path):
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                data = json.loads(line)
                dataset.append({
                    'query': data['query'],
                    'A': data['choices'][0],
                    'B': data['choices'][1],
                    'C': data['choices'][2],
                    'D': data['choices'][3],
                    'gold': data['gold'],
                })
        dataset = Dataset.from_list(dataset)
        return dataset
