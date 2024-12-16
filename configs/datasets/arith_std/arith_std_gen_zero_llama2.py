from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import ArithDataset,ArithEvaluator,arith_std_first_option_postprocess
from opencompass.utils.text_postprocessors import math_answer_extract
import os


arith_reader_cfg = dict(input_columns=['expression'], output_column='answer')


arith_all_sets = os.listdir('./data/arith_std_3K/')
# arith_all_sets = ['low_digits_standard.json']

arith_std_datasets = []
for _name in arith_all_sets:
    #print('*'*20,_name)
    
    _name = _name.split('.')[0]
    arith_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[

                dict(role='HUMAN', prompt="[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n\n Calculate the following arithmetic problem: {expression} [/INST] "),

            ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=2048))

    arith_eval_cfg = dict(
        evaluator=dict(type=ArithEvaluator), 
        pred_role="BOT",
        pred_postprocessor=dict(type=math_answer_extract)
        )
    # if _name != 'mixed_compound_standard':
    arith_std_datasets.append(
        dict(
            type=ArithDataset,
            abbr=f'arith_std_{_name}',
            path=f'./data/arith_std_3K/',
            name=_name,
            reader_cfg=arith_reader_cfg,
            infer_cfg=arith_infer_cfg,
            eval_cfg=arith_eval_cfg)
    )

del _name