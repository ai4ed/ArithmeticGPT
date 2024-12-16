from os.path import exists
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
# from opencompass.datasets import BBHDataset, BBHEvaluator, bbh_mcq_postprocess
from opencompass.datasets import BBArithmeticDataset, BBEvaluator, bb_mcq_postprocess
from opencompass.utils.text_postprocessors import merge_first_option_postprocess4all,math_answer_extract
bbh_reader_cfg = dict(input_columns=["input"], output_column="target")

_path_prefix = "./data/bb_arithmetic"

# bbh_multiple_choice_sets = [
#     'temporal_sequences',
#     'disambiguation_qa',
#     'date_understanding',
#     'tracking_shuffled_objects_three_objects',
#     'penguins_in_a_table',
#     'geometric_shapes',
#     'snarks',
#     'ruin_names',
#     'tracking_shuffled_objects_seven_objects',
#     'tracking_shuffled_objects_five_objects',
#     'logical_deduction_three_objects',
#     'hyperbaton',
#     'logical_deduction_five_objects',
#     'logical_deduction_seven_objects',
#     'movie_recommendation',
#     'salient_translation_error_detection',
#     'head',
# ]
bbh_free_form_sets = [
    '1_digit_addition',
    '1_digit_division',
    '1_digit_multiplication',
    '1_digit_subtraction',
    
    '2_digit_addition',
    '2_digit_division',
    '2_digit_multiplication',
    '2_digit_subtraction',
    
    '3_digit_addition',
    '3_digit_division',
    '3_digit_multiplication',
    '3_digit_subtraction',
    
    '4_digit_addition',
    '4_digit_division',
    '4_digit_multiplication',
    '4_digit_subtraction',
    
    '5_digit_addition',
    '5_digit_division',
    '5_digit_multiplication',
    '5_digit_subtraction',
    
    
]

bb_arith_datasets = []


for _name in bbh_free_form_sets:
    bbh_infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(round=[
                
                 dict(role='HUMAN', prompt="<reserved_106> Here is a math question, you need answer the question. Question:{input}\nThe correct answer is <reserved_107>"),
            ])),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer, max_out_len=2048))
    bbh_eval_cfg = dict(evaluator=dict(type=BBEvaluator),
                        pred_role="BOT",
                        pred_postprocessor=dict(type=math_answer_extract)
                        )

    bb_arith_datasets.append(
        dict(
            type=BBArithmeticDataset,
            path=f"{_path_prefix}/",
            name=_name,
            abbr='BBArithmetic-' + _name,
            reader_cfg=bbh_reader_cfg,
            infer_cfg=bbh_infer_cfg.copy(),
            eval_cfg=bbh_eval_cfg.copy()))

del _name, _path_prefix
