import os
import json
from datasets import Dataset
from datasets import DatasetDict, load_dataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import  (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)


from .base import BaseDataset
import re

@LOAD_DATASET.register_module()
class ArithDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        raw_data = []

        if '200' in path:
            file_path = path
        else:
            file_path = os.path.join(path, f'{name}.json')
        print('file_path--', file_path)
        with open(file_path, 'r',encoding='utf-8') as fr:
            reader = json.load(fr)

            new_reader = []
            for sample in reader:
                new_sample = {}
                new_sample['id'] = sample['id']
                new_sample['expression'] = sample['expression']
                new_sample['answer'] = str(sample['answer'])
                new_reader.append(new_sample)
            raw_data.extend(new_reader)
        dataset = Dataset.from_list(raw_data)
        # print('len(dataset)--', len(dataset))
        return dataset

@TEXT_POSTPROCESSORS.register_module('arith_dataset')
def arith_std_first_option_postprocess(text: str) -> str:
    """Find first valid option for text."""

    patterns = [
        # r"####\s?(\-?\d+[\.|\,]?\d*i*)\s?",
        r"answer is (\-?\d+[\.|\,]?\d*i*)",
        r"答(\-?\d+[\.|\,]?\d*i*)",
        r"因此(\-?\d+[\.|\,]?\d*i*)",
        r"所以(\-?\d+[\.|\,]?\d*i*)",
        r"so (\-?\d+[\.|\,]?\d*i*)",
        r"Therefore (\-?\d+[\.|\,]?\d*i*)",
        r"=> (\-?\d+[\.|\,]?\d*i*)",
        r'=\s*(\-?\d+[\.|\,]?\d*i*)\n',
        r"(\-?\d+[\.|\,]?\d*i*)\n",
        r"(\-?\d+[\.|\,]?\d*i*)",
    ]

    regexes = [re.compile(pattern) for pattern in patterns]
    for regex in regexes:
        match = regex.findall(text)
        print('match: ', match)
        if match:
            outputs = match[-1]
            outputs = outputs.replace(',', '').strip().strip('\n')
            if outputs.endswith('.'):
                outputs = outputs+'0'
            print('outputs: ', outputs)
            return outputs
    return ''


@ICL_EVALUATORS.register_module()
class ArithEvaluator(BaseEvaluator):

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        correct = 0
        count = 0
        for i, j in zip(predictions, references):
            count += 1
            if self.is_equiv(i, j):
                correct += 1
        result = {'accuracy': 100 * correct / count}
        return result

    def is_equiv(self, str1, str2, verbose=False):
        
        if str1 is None and str2 is None:
            print('WARNING: Both None')
            return True
        if str1 is None or str2 is None:
            return False

        if not isinstance(str1, str):
            str1 = str(str1)
        if not isinstance(str2, str):
            str2 = str(str2)
            
        if ('i' in str1 and 'i' not in str2) or ('i' in str2 and 'i' not in str1):
            return False
        if 'i' in str1 and 'i' in str2:
            str1 = str1.strip('i')
            str2 = str2.strip('i')
   
        try:
            pred = float(str1)
            label = float(str2)
            return abs(pred - label) <= 1e-3
        except:  # noqa
            return False
