from opencompass.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)
import json
from .base import BaseDataset
from datasets import Dataset, DatasetDict
import re
from opencompass.utils.text_postprocessors import answer_postprocess


@LOAD_DATASET.register_module()
class GSM8KDataset(BaseDataset):

    @staticmethod
    def load(path: str):

        dataset = DatasetDict()
        for dn in ['test', 'train']:
            datalist =  []
            datapath = path + dn + '.jsonl'
            print("datapath : ", datapath)
            with open(datapath, 'r') as json_file:
                for line in json_file:
                    data = json.loads(line)
                    datalist.append(data)

            raw_data = []
            for item in datalist:
                raw_data.append({
                    'question':
                    item['question'],
                    'answer':
                    item['answer']
                })           
            dataset[dn] = Dataset.from_list(raw_data)
        return dataset
    
@TEXT_POSTPROCESSORS.register_module('gsm8k_dataset')
def gsm8k_dataset_postprocess(text: str) -> str:
    return text.split('#### ')[1].replace(',', '')


@TEXT_POSTPROCESSORS.register_module('gsm8k')

def gsm8k_postprocess(text: str) -> str:
    matches = answer_postprocess(text)
    numbers = re.findall(r'[-+]?\d+(?:,\d+)?(?:\.\d+)?', str(matches))
    if len(numbers) == 0:
        return text
    text = numbers[-1].strip().strip('.,?!\"\';:')
    text = text.replace(",", '')     
    text = str(text)
    # print("2222----text is : ", text)
    return text