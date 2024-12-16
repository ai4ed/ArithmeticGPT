from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import hellaswagDataset_V1
from opencompass.utils.text_postprocessors import first_option_postprocess

hellaswag_reader_cfg = dict(
    input_columns=["ctx", "A", "B", "C", "D"],
    output_column="label"
    # test_split="validation"
    )

hellaswag_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role="HUMAN",
                # prompt=("{ctx}\nQuestion: Which ending makes the most sense?\n"
                #         "A. {A}\nB. {B}\nC. {C}\nD. {D}\n"
                #         "You may choose from 'A', 'B', 'C', 'D'.\n"
                #         "Answer:"),
                prompt = "[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n\n \
{ctx}\nWhich ending makes the most sense?\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nYou may choose from 'A', 'B', 'C', 'D'. [/INST] "
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever, ),
    inferencer=dict(type=GenInferencer,  max_out_len=2048),
)

hellaswag_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)

hellaswag_datasets = [
    dict(
        abbr='hellaswag',
        type=hellaswagDataset_V1,
        path="./data/hellaswag/hellaswag.jsonl",
        reader_cfg=hellaswag_reader_cfg,
        infer_cfg=hellaswag_infer_cfg,
        eval_cfg=hellaswag_eval_cfg)
]
