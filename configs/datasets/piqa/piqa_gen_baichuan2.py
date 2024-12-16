from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import piqaDataset_V1
from opencompass.utils.text_postprocessors import first_option_postprocess

piqa_reader_cfg = dict(
    input_columns=["goal", "sol1", "sol2"],
    output_column="answer",
    )

piqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role="HUMAN",
                    # prompt="{goal}\nA. {sol1}\nB. {sol2}\nAnswer:"
                    prompt = "<reserved_106>{goal}\nA. {sol1}\nB. {sol2}\n<reserved_107>"
                    )
            ], ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer,  max_out_len=2048),
)

piqa_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
)

piqa_datasets = [
    dict(
        bbr="piqa",
        type=piqaDataset_V1,
        path="./data/PIQA",
        reader_cfg=piqa_reader_cfg,
        infer_cfg=piqa_infer_cfg,
        eval_cfg=piqa_eval_cfg)
]
