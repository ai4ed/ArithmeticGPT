from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import AsdivDataset,asdiv_first_option_postprocess, asdiv_postprocess, last_answer_postprocess

import os


asdiv_reader_cfg = dict(input_columns=['problem'], output_column='answer')


asdiv_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
           
            dict(role='HUMAN', prompt="[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n\n Calculate the following math word problem: {problem}\nAnswer: [/INST] "),
            # dict(role='HUMAN', prompt="[INST] Calculate the following math word problem: {problem}\nAnswer: [/INST]"),

        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048))

asdiv_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator), 
    pred_role="BOT",
    pred_postprocessor=dict(type=asdiv_postprocess)
    )

asdiv_datasets = [
    dict(
        type=AsdivDataset,
        abbr='asdiv-a',
        path='./data/asdiv-a/',
        name='asdiv-a',
        reader_cfg=asdiv_reader_cfg,
        infer_cfg=asdiv_infer_cfg,
        eval_cfg=asdiv_eval_cfg)
]

