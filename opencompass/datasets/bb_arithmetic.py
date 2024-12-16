import json
import os.path as osp
import re

from datasets import Dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import (ICL_EVALUATORS, LOAD_DATASET,
                                  TEXT_POSTPROCESSORS)

from .base import BaseDataset
from opencompass.utils.text_postprocessors import merge_first_option_postprocess4all

@LOAD_DATASET.register_module()
class BBArithmeticDataset(BaseDataset):

    @staticmethod
    def load(path: str, name: str):
        with open(osp.join(path, name,'task.json'), 'r') as f:
            data = json.load(f)['examples']
        dataset = Dataset.from_list(data)
        return dataset


@TEXT_POSTPROCESSORS.register_module('bb-mcq')
def bb_mcq_postprocess(text: str) -> str:
    ans = text
    ans_line = ans.split('answer is ')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    match = re.search(r'\(([A-Z])\)*', ans)
    if match:
        return match.group(1)
    match = re.search(r'([A-Z])', ans)
    if match:
        return match.group(1)
    return ans


@TEXT_POSTPROCESSORS.register_module('bb-freeform')
def bb_freeform_postprocess(text: str) -> str:
    ans = text
    ans_line = ans.split('answer is ')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    ans = ans.split('\n')[0]
    if ans.endswith('.'):
        ans = ans[:-1]
    return ans

 


@ICL_EVALUATORS.register_module()
class BBEvaluator(BaseEvaluator):
    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        predictions = [merge_first_option_postprocess4all(pred,options=r'-?\d+\.\d+|-?\d+/?\d*') for pred in predictions]
        print('==BBEvaluator=',list(zip(references,predictions)))
        cnt = 0
        for pred, ref in zip(predictions, references):
            if pred == ref:
                cnt += 1

        score = cnt / len(predictions) * 100

        return {'score': score}
