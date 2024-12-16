from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import GSM8KDataset, gsm8k_postprocess, gsm8k_dataset_postprocess

gsm8k_reader_cfg = dict(input_columns=['question'], output_column='answer')

gsm8k_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        # template=dict(
        #     round=[
        #         dict(role='HUMAN', prompt="Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?\nLet's think step by step\nAnswer:"),
        #         dict(role='BOT', prompt="Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.\nFor the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.\nAngelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.\nHowever, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.\nThey also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.\nAnd they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.\nSo Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.\nThey want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75\nThey will need to plan to study 4 days to allow for all the time they need.\nThe answer is 4\n"),
        #         # dict(role='HUMAN', prompt="Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?\nLet's think step by step\nAnswer:"),
        #         # dict(role='BOT', prompt="Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.\nHis team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers\nThey scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.\nAll together his team scored 50+24+10= 84 points\nMark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.\nHis opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.\nThey also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.\nAll together Mark's opponents scored 100+12+5=117 points\nThe total score for the game is both team's scores added together, so it is 84+117=201 points\nThe answer is 201\n"),
        #         # dict(role='HUMAN', prompt="Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?\nLet's think step by step\nAnswer:"),
        #         # dict(role='BOT', prompt="When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24\nThe total number of marbles she'll have is 60+24 = 84\nIf Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.\nIf Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.\nThe total number of frisbees she'll have will increase to 30+12 = 42\nBella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards\nIf she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.\nThe total number of deck cards she'll have is 10+4 = 14\nTogether, Bella will have a total of 14+42+84 = 140 items\nThe answer is 140\n"),
        #         # dict(role='HUMAN', prompt="Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?\nLet's think step by step\nAnswer:"),
        #         # dict(role='BOT', prompt="For the first three baskets, the number of apples and oranges in one basket is 9+15=24\nIn total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.\nSince there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.\nThe number of apples in the fourth basket is 9-2=7\nThere are also 15-2=13 oranges in the fourth basket\nThe combined number of oranges and apples in the fourth basket is 13+7=20\nThe fourth basket also contains 14-2=12 bananas.\nIn total, the fourth basket has 20+12=32 fruits.\nThe four baskets together have 32+114=146 fruits.\nThe answer is 146\n"),
        #         dict(role='HUMAN', prompt="Question: {question}\nLet's think step by step\nAnswer:"),
        #     ],
        # )
        template="<reserved_106>一桶奶粉上标有净含量$\\left( 900\\pm 5 \\right)$克，这种奶粉最轻不少于（   ）克．A. $900$B. $905$C. $895$D. $890$\
<reserved_107>这道题目需要我们理解净含量的概念。净含量$\\left( 900\\pm 5 \\right)$克表示这种奶粉的净含量在$900$克左右，误差范围为$5$克。因此，这种奶粉的净含量最少是$900-5=895$克，最多是$900+5=905$克。题目要求这种奶粉最轻不少于多少克，因此我们需要选择最小的净含量，即$895$克。因此，答案是C。\
<reserved_106>你能详细给我解答一下 计算：$1\\times 2+2\\times 3+3\\times 4+\\cdots +14\\times 15$ 这个问题的解题步骤吗？\
<reserved_107>这道题目是一个求和问题，我们可以使用整数拆分法来解决它。原式可以写成$1\\times 2+2\\times 3+3\\times 4+\\cdots +14\\times 15$。我们可以将每一项拆分成两个部分，例如第一项可以拆分成$1\\times 2=2-1$，第二项可以拆分成$2\\times 3=6-2$，以此类推。这样，原式就可以写成$2-1+6-2+12-3+\\cdots +210-14$。我们可以将这些项分组，每组两项，例如第一组是$2-1+6-2=3+4$，第二组是$12-3+20-4=9+16$，以此类推。这样，原式就可以写成$\\left( 3+4 \\right)+\\left( 9+16 \\right)+\\cdots +\\left( 195+196 \\right)+\\left( 210-14 \\right)$。我们可以将每组两项相加，例如第一组是$3+4=7$，第二组是$9+16=25$，以此类推。这样，原式就可以写成$7+25+\\cdots +401+196$。我们可以使用等差数列求和公式来计算这个和，其中首项是$7$，公差是$18$，项数是$14$。因此，原式$=\\frac{14}{2}\\times \\left( 7+196 \\right)=$ <thought>这里需要计算乘法和除法，我需要在这里调用 Calculator API <API>[{\"ActionName\": \"Calculator\", \"Args\": {\"equation\":\"14/2*(7+196)\"}}] </API> => 1120</thought> $1120$。因此，原式的值是1120。\
<reserved_106>一条乡间小道长$600$米，在这条小道的两侧等距离栽种$302$棵美人蕉（两端要栽），每两棵美人蕉相距多少米？ 这个问题你能帮我解答一下吗\
<reserved_107>这条小道的长度是 600 米，两侧等距离栽种了 302 棵美人蕉。因为两端也要栽种，所以每侧种了 $302\\div 2=$<thought>这里需要计算除法，我需要在这里调用 Calculator API<API>[{\"ActionName\": \"Calculator\", \"Args\": {\"equation\": \"302/2\"}}]</API> => 151</thought> 151 棵美人蕉。这些美人蕉将小道分成了 150 个等长的区间。因此，每两棵美人蕉之间的距离是 $600\\div 150=$<thought>这里需要计算除法，我需要在这里调用 Calculator API<API>[{\"ActionName\": \"Calculator\", \"Args\": {\"equation\": \"600/150\"}}]</API> => 4</thought> 4 米。因此，每两棵美人蕉之间相距4米。\
<reserved_106>{question}\
<reserved_107>"
        ),
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
        abbr='gsm8k-few',
        type=GSM8KDataset,
        path='./data/GSM8K/',
        reader_cfg=gsm8k_reader_cfg,
        infer_cfg=gsm8k_infer_cfg,
        eval_cfg=gsm8k_eval_cfg)
]