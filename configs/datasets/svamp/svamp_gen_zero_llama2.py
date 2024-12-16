from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
# from opencompass.datasets import HFDataset, svamp_postprocess, svamp_dataset_postprocess
from opencompass.datasets import SVAMPDataset
from opencompass.utils.text_postprocessors import one_answer_extract


svamp_reader_cfg = dict(input_columns=['problem'], output_column='answer')

# _hint = 'The answer is '
svamp_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                # dict(role='HUMAN', prompt="Calculate the following math word problem: {problem}\nPlease give the answer directly\nAnswer:"),
                dict(role='HUMAN', prompt="[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n\n Calculate the following math word problem: {problem}\nAnswer: [/INST] "),
                # dict(role='HUMAN', prompt="[INST] Calculate the following math word problem: {problem}\nAnswer: [/INST]"),
                
                ]
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048))


# svamp_eval_cfg = dict(evaluator=dict(type=MATHEvaluator),
#                       pred_postprocessor=dict(type=math_postprocess),
#                      )

svamp_eval_cfg = dict(evaluator=dict(type=AccEvaluator),
                      pred_postprocessor=dict(type=one_answer_extract),
                     )

svamp_datasets = [
    dict(
        abbr='svamp',
        type=SVAMPDataset,
        path='./data/svamp/svamp_test.jsonl',
        reader_cfg=svamp_reader_cfg,
        infer_cfg=svamp_infer_cfg,
        eval_cfg=svamp_eval_cfg)
]

