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
               
                dict(role='HUMAN', prompt="[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n\n Calculate the following math word problem: {question}\nAnswer: [/INST] "),
            ],
            # round=[
               
                # dict(role='HUMAN', prompt="[INST] Calculate the following math word problem: {question}\nAnswer: [/INST]"),
            # ],
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