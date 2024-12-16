from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import Math401Dataset
from opencompass.utils.text_postprocessors import one_answer_extract, math_answer_extract


math401_reader_cfg = dict(input_columns=['problem'], output_column='answer')

# _hint = 'The answer is '
math401_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        # template=dict(
        #     round=[
        #         dict(role='HUMAN', prompt="Problem: 5+2=\nAnswer:"),
        #         dict(role='BOT', prompt="The answer is 7\n"),
        #         # dict(role='HUMAN', prompt="Problem: log 10(563)=\nAnswer:"),
        #         # dict(role='BOT', prompt="The answer is 2.7505\n"),
        #         # dict(role='HUMAN', prompt="Problem: 45*61=\nAnswer:"),
        #         # dict(role='BOT', prompt="The answer is 2745\n"),
        #         dict(role='HUMAN', prompt="Problem:{problem}\nAnswer:")
        #         ]
        # )
#         template ="[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\
# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don\'t know the answer to a question, please don\'t share false information.\n<</SYS>>\n\n一桶奶粉上标有净含量$\\left( 900\\pm 5 \\right)$克，这种奶粉最轻不少于（   ）克．A. $900$B. $905$C. $895$D. $890$ [/INST] \
# 这道题目需要我们理解净含量的概念。净含量$\\left( 900\\pm 5 \\right)$克表示这种奶粉的净含量在$900$克左右，误差范围为$5$克。因此，这种奶粉的净含量最少是$900-5=895$克，最多是$900+5=905$克。题目要求这种奶粉最轻不少于多少克，因此我们需要选择最小的净含量，即$895$克。因此，答案是C。 </s><s>\
# [INST] 你能详细给我解答一下 计算：$1\\times 2+2\\times 3+3\\times 4+\\cdots +14\\times 15$ 这个问题的解题步骤吗？ [/INST] \
# 这道题目是一个求和问题，我们可以使用整数拆分法来解决它。原式可以写成$1\\times 2+2\\times 3+3\\times 4+\\cdots +14\\times 15$。我们可以将每一项拆分成两个部分，例如第一项可以拆分成$1\\times 2=2-1$，第二项可以拆分成$2\\times 3=6-2$，以此类推。这样，原式就可以写成$2-1+6-2+12-3+\\cdots +210-14$。我们可以将这些项分组，每组两项，例如第一组是$2-1+6-2=3+4$，第二组是$12-3+20-4=9+16$，以此类推。这样，原式就可以写成$\\left( 3+4 \\right)+\\left( 9+16 \\right)+\\cdots +\\left( 195+196 \\right)+\\left( 210-14 \\right)$。我们可以将每组两项相加，例如第一组是$3+4=7$，第二组是$9+16=25$，以此类推。这样，原式就可以写成$7+25+\\cdots +401+196$。我们可以使用等差数列求和公式来计算这个和，其中首项是$7$，公差是$18$，项数是$14$。因此，原式$=\\frac{14}{2}\\times \\left( 7+196 \\right)=$ <thought>这里需要计算乘法和除法，我需要在这里调用 Calculator API <API>[{\"ActionName\": \"Calculator\", \"Args\": {\"equation\":\"14/2*(7+196)\"}}] </API> => 1120</thought> $1120$。因此，原式的值是1120。 </s><s>\
# [INST] 一条乡间小道长$600$米，在这条小道的两侧等距离栽种$302$棵美人蕉（两端要栽），每两棵美人蕉相距多少米？ 这个问题你能帮我解答一下吗？ [/INST] \
# 这条小道的长度是 600 米，两侧等距离栽种了 302 棵美人蕉。因为两端也要栽种，所以每侧种了 $302\\div 2=$<thought>这里需要计算除法，我需要在这里调用 Calculator API<API>[{\"ActionName\": \"Calculator\", \"Args\": {\"equation\": \"302/2\"}}]</API> => 151</thought> 151 棵美人蕉。这些美人蕉将小道分成了 150 个等长的区间。因此，每两棵美人蕉之间的距离是 $600\\div 150=$<thought>这里需要计算除法，我需要在这里调用 Calculator API<API>[{\"ActionName\": \"Calculator\", \"Args\": {\"equation\": \"600/150\"}}]</API> => 4</thought> 4 米。因此，每两棵美人蕉之间相距4米。 </s><s>\
# [INST] {problem} [/INST] "
        template="[INST] Please answer the following question.\nQuestion: 5+2=\nAnswer: \
[/INST] 5+2=的结果是 <thought> 涉及到计算，我需要在这里调用 Calculator API <API> [{\"ActionName\": \"Calculator\", \"Args\": {\"equation\": \"5+2=\"}}] </API> => 7 </thought> 7  </s><s>\
[INST] Question: {problem}\nAnswer: [/INST] "
        ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048))



math401_eval_cfg = dict(evaluator=dict(type=AccEvaluator),
                      pred_postprocessor=dict(type=one_answer_extract),
                     )

math401_datasets = [
    dict(
        abbr='math401-few',
        type=Math401Dataset,
        path='./data/math401-llm/math401.json',
        reader_cfg=math401_reader_cfg,
        infer_cfg=math401_infer_cfg,
        eval_cfg=math401_eval_cfg)
]

