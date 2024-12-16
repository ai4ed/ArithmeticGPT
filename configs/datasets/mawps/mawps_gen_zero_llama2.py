from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import MawpsDataset,first_option_postprocess, mawps_postprocess
from opencompass.utils.text_postprocessors import first_capital_postprocess


math_reader_cfg = dict(input_columns=['original_text'], output_column='ans')

math_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[

            dict(role='HUMAN', prompt="[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n\n Calculate the following math word problem: {original_text}\nAnswer: [/INST] "),
            # dict(role='HUMAN', prompt="[INST] Calculate the following math word problem: {original_text}\nAnswer: [/INST]"),

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