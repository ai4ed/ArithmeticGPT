from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import Math401Dataset
from opencompass.utils.text_postprocessors import one_answer_extract



math401_reader_cfg = dict(input_columns=['problem'], output_column='answer')

# _hint = 'The answer is '
math401_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                # dict(role='HUMAN', prompt="Calculate the following math problem: {problem}\nPlease give the answer directly\nAnswer:"),
                dict(role='HUMAN', prompt="计算下面的数学题: {problem}\n请直接给出答案\n答:"),
                ]
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))



math401_eval_cfg = dict(evaluator=dict(type=AccEvaluator),
                      pred_postprocessor=dict(type=one_answer_extract),
                     )

math401_datasets = [
    dict(
        abbr='math401',
        type=Math401Dataset,
        path='./data/math401-llm/math401.json',
        reader_cfg=math401_reader_cfg,
        infer_cfg=math401_infer_cfg,
        eval_cfg=math401_eval_cfg)
]

