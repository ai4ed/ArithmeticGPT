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
                # dict(role='HUMAN', prompt="Problem: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nAnswer:"),
                # dict(role='BOT', prompt="The answer is 4.0\n"),
                dict(role='HUMAN', prompt="Problem: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\nAnswer:"),
                dict(role='BOT', prompt="The answer is 201.0\n"),
                # dict(role='HUMAN', prompt="Problem: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\nAnswer:"),
                # dict(role='BOT', prompt="The answer is 140\n"),
                # dict(role='HUMAN', prompt="Problem: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\nAnswer:"),
                # dict(role='BOT', prompt="The answer is 146.0\n"),
                dict(role='HUMAN', prompt="Problem:{problem}\nAnswer:")
                ]
   
            
        )),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=512))


# svamp_eval_cfg = dict(evaluator=dict(type=MATHEvaluator),
#                       pred_postprocessor=dict(type=math_postprocess),
#                      )

svamp_eval_cfg = dict(evaluator=dict(type=AccEvaluator),
                      pred_postprocessor=dict(type=one_answer_extract),
                     )

svamp_datasets = [
    dict(
        abbr='svamp-few',
        type=SVAMPDataset,
        path='./data/svamp/svamp_test.jsonl',
        reader_cfg=svamp_reader_cfg,
        infer_cfg=svamp_infer_cfg,
        eval_cfg=svamp_eval_cfg)
]

