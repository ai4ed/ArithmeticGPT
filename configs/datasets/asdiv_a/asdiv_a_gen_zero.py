from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import AsdivDataset,asdiv_first_option_postprocess
import os


asdiv_reader_cfg = dict(input_columns=['problem'], output_column='answer')


asdiv_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role="HUMAN",
                prompt=
                "Calculate the following math word problem: {problem}\n Answer:"
            ),

        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))

asdiv_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator), 
    pred_role="BOT",
    pred_postprocessor=dict(type=asdiv_first_option_postprocess)
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

