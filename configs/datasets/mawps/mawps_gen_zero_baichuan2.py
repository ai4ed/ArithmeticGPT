from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MawpsDataset,first_option_postprocess,mawps_postprocess
from opencompass.utils.text_postprocessors import first_capital_postprocess
from opencompass.datasets import  MATHEvaluator, math_postprocess

math_reader_cfg = dict(input_columns=['original_text'], output_column='ans')

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            # dict(
            #     role="HUMAN",
            #     prompt=
            #     "Calculate the following math word problem: {original_text}\n Answer:"
            # ),
             dict(role='HUMAN', prompt="<reserved_106> Calculate the following math word problem: {original_text}\nAnswer: <reserved_107>"),

        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048))

math_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator), 
    pred_role="BOT",
    pred_postprocessor=dict(type=mawps_postprocess)
    )

mawps_datasets = [
    dict(
        type=MawpsDataset,
        abbr='mawps',
        path='./data/mawps/',
        reader_cfg=math_reader_cfg,
        infer_cfg=math_infer_cfg,
        eval_cfg=math_eval_cfg)
]