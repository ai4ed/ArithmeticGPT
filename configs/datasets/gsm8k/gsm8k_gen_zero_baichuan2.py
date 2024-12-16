from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
               
                dict(role='HUMAN', prompt="<reserved_106> Calculate the following math word problem: {question}\nAnswer: <reserved_107>"),
            ],
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048))# 512

gsm8k_eval_cfg = dict(evaluator=dict(type=AccEvaluator),
                      pred_postprocessor=dict(type=gsm8k_postprocess),
                      dataset_postprocessor=dict(type=gsm8k_dataset_postprocess))

# gsm8k_datasets = [
#     dict(
#         abbr='gsm8k',
#         type=GSM8KDataset,
#         path='./data/GSM8K/',
#         name='main',
#         reader_cfg=gsm8k_reader_cfg,
#         infer_cfg=gsm8k_infer_cfg,
#         eval_cfg=gsm8k_eval_cfg)
# ]
gsm8k_datasets = [
    dict(
        abbr='gsm8k',
        type=GSM8KDataset,
        path='./data/GSM8K/',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]