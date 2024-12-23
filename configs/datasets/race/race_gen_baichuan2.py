from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import RaceDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

race_reader_cfg = dict(
    input_columns=['article', 'question', 'A', 'B', 'C', 'D'],
    output_column='answer')

race_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role="HUMAN",
                # prompt=
                # "Read the article, and answer the question by replying A, B, C or D.\n\nArticle:\n{article}\n\nQ: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}"
                prompt = "<reserved_106>Read the article, and answer the question by replying A, B, C or D.\n\nArticle:\n{article}\n\nQuestion: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}<reserved_107>"
            ),
        ])),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer,  max_out_len=2048))

race_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
    pred_role='BOT')

race_datasets = [
    dict(
        type=RaceDataset,
        abbr='race-middle',
        path='race',
        name='middle',
        reader_cfg=race_reader_cfg,
        infer_cfg=race_infer_cfg,
        eval_cfg=race_eval_cfg),
    dict(
        type=RaceDataset,
        abbr='race-high',
        path='race',
        name='high',
        reader_cfg=race_reader_cfg,
        infer_cfg=race_infer_cfg,
        eval_cfg=race_eval_cfg)
]
