from datasets import load_dataset, Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .base import BaseDataset
import json
import os

@LOAD_DATASET.register_module()
class RaceDataset(BaseDataset):


    @staticmethod
    def load(path: str, name: str):
        # dataset = load_datasets(path, name)
        dataset = DatasetDict()
        datapath = path + '/' + name + '.json'
        # print('datapath: ', datapath)
        with open(datapath, 'r') as json_file:
                datalist = json.load(json_file)
        
        raw_data = []
        for item in datalist:
            for i in range(len(item['questions'])):
                raw_data.append({
                    'question': item['questions'][i],
                    'article': item['article'],
                    'A': item['options'][i][0],
                    'B': item['options'][i][1],
                    'C': item['options'][i][2],
                    'D': item['options'][i][3],
                    'answer':item['answers'][i]
                })
            # print('raw_data:', raw_data[-1])           
        dataset['test'] = Dataset.from_list(raw_data)
        dataset['train'] = Dataset.from_list(raw_data)
        return dataset
