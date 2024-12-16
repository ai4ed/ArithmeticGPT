from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever,FixKRetriever
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import ArithDataset,ArithEvaluator,arith_std_first_option_postprocess
import os


arith_reader_cfg = dict(input_columns=['expression'], output_column='answer')


arith_all_sets = os.listdir('./data/arith_std/')
# arith_all_sets = ['low_digits_standard.json']

arith_std_datasets_few = []
for _name in arith_all_sets:
    #print('*'*20,_name)
    
    _name = _name.split('.')[0]
    arith_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(
                    begin="</E>",
                    round=[
                    dict(
                        role='HUMAN',
                        prompt="<reserved_106>一桶奶粉上标有净含量$\\left( 900\\pm 5 \\right)$克，这种奶粉最轻不少于（   ）克．A. $900$B. $905$C. $895$D. $890$\
<reserved_107>这道题目需要我们理解净含量的概念。净含量$\\left( 900\\pm 5 \\right)$克表示这种奶粉的净含量在$900$克左右，误差范围为$5$克。因此，这种奶粉的净含量最少是$900-5=895$克，最多是$900+5=905$克。题目要求这种奶粉最轻不少于多少克，因此我们需要选择最小的净含量，即$895$克。因此，答案是C。\
<reserved_106>你能详细给我解答一下 计算：$1\\times 2+2\\times 3+3\\times 4+\\cdots +14\\times 15$ 这个问题的解题步骤吗？\
<reserved_107>这道题目是一个求和问题，我们可以使用整数拆分法来解决它。原式可以写成$1\\times 2+2\\times 3+3\\times 4+\\cdots +14\\times 15$。我们可以将每一项拆分成两个部分，例如第一项可以拆分成$1\\times 2=2-1$，第二项可以拆分成$2\\times 3=6-2$，以此类推。这样，原式就可以写成$2-1+6-2+12-3+\\cdots +210-14$。我们可以将这些项分组，每组两项，例如第一组是$2-1+6-2=3+4$，第二组是$12-3+20-4=9+16$，以此类推。这样，原式就可以写成$\\left( 3+4 \\right)+\\left( 9+16 \\right)+\\cdots +\\left( 195+196 \\right)+\\left( 210-14 \\right)$。我们可以将每组两项相加，例如第一组是$3+4=7$，第二组是$9+16=25$，以此类推。这样，原式就可以写成$7+25+\\cdots +401+196$。我们可以使用等差数列求和公式来计算这个和，其中首项是$7$，公差是$18$，项数是$14$。因此，原式$=\\frac{14}{2}\\times \\left( 7+196 \\right)=$ <thought>这里需要计算乘法和除法，我需要在这里调用 Calculator API <API>[{\"ActionName\": \"Calculator\", \"Args\": {\"equation\":\"14/2*(7+196)\"}}] </API> => 1120</thought> $1120$。因此，原式的值是1120。\
<reserved_106>一条乡间小道长$600$米，在这条小道的两侧等距离栽种$302$棵美人蕉（两端要栽），每两棵美人蕉相距多少米？ 这个问题你能帮我解答一下吗\
<reserved_107>这条小道的长度是 600 米，两侧等距离栽种了 302 棵美人蕉。因为两端也要栽种，所以每侧种了 $302\\div 2=$<thought>这里需要计算除法，我需要在这里调用 Calculator API<API>[{\"ActionName\": \"Calculator\", \"Args\": {\"equation\": \"302/2\"}}]</API> => 151</thought> 151 棵美人蕉。这些美人蕉将小道分成了 150 个等长的区间。因此，每两棵美人蕉之间的距离是 $600\\div 150=$<thought>这里需要计算除法，我需要在这里调用 Calculator API<API>[{\"ActionName\": \"Calculator\", \"Args\": {\"equation\": \"600/150\"}}]</API> => 4</thought> 4 米。因此，每两棵美人蕉之间相距4米。\
<reserved_106>{expression}\
<reserved_107>"
                    ),
                    dict(role='BOT', prompt='{answer}\n')
                ]),
            ice_token="</E>"
            ), 
        prompt_template=dict(
            type=PromptTemplate,
            template=dict(
                    begin="</E>",
                    round=[
                    dict(
                        role='HUMAN',
                        prompt="<reserved_106>一桶奶粉上标有净含量$\\left( 900\\pm 5 \\right)$克，这种奶粉最轻不少于（   ）克．A. $900$B. $905$C. $895$D. $890$\
<reserved_107>这道题目需要我们理解净含量的概念。净含量$\\left( 900\\pm 5 \\right)$克表示这种奶粉的净含量在$900$克左右，误差范围为$5$克。因此，这种奶粉的净含量最少是$900-5=895$克，最多是$900+5=905$克。题目要求这种奶粉最轻不少于多少克，因此我们需要选择最小的净含量，即$895$克。因此，答案是C。\
<reserved_106>你能详细给我解答一下 计算：$1\\times 2+2\\times 3+3\\times 4+\\cdots +14\\times 15$ 这个问题的解题步骤吗？\
<reserved_107>这道题目是一个求和问题，我们可以使用整数拆分法来解决它。原式可以写成$1\\times 2+2\\times 3+3\\times 4+\\cdots +14\\times 15$。我们可以将每一项拆分成两个部分，例如第一项可以拆分成$1\\times 2=2-1$，第二项可以拆分成$2\\times 3=6-2$，以此类推。这样，原式就可以写成$2-1+6-2+12-3+\\cdots +210-14$。我们可以将这些项分组，每组两项，例如第一组是$2-1+6-2=3+4$，第二组是$12-3+20-4=9+16$，以此类推。这样，原式就可以写成$\\left( 3+4 \\right)+\\left( 9+16 \\right)+\\cdots +\\left( 195+196 \\right)+\\left( 210-14 \\right)$。我们可以将每组两项相加，例如第一组是$3+4=7$，第二组是$9+16=25$，以此类推。这样，原式就可以写成$7+25+\\cdots +401+196$。我们可以使用等差数列求和公式来计算这个和，其中首项是$7$，公差是$18$，项数是$14$。因此，原式$=\\frac{14}{2}\\times \\left( 7+196 \\right)=$ <thought>这里需要计算乘法和除法，我需要在这里调用 Calculator API <API>[{\"ActionName\": \"Calculator\", \"Args\": {\"equation\":\"14/2*(7+196)\"}}] </API> => 1120</thought> $1120$。因此，原式的值是1120。\
<reserved_106>一条乡间小道长$600$米，在这条小道的两侧等距离栽种$302$棵美人蕉（两端要栽），每两棵美人蕉相距多少米？ 这个问题你能帮我解答一下吗\
<reserved_107>这条小道的长度是 600 米，两侧等距离栽种了 302 棵美人蕉。因为两端也要栽种，所以每侧种了 $302\\div 2=$<thought>这里需要计算除法，我需要在这里调用 Calculator API<API>[{\"ActionName\": \"Calculator\", \"Args\": {\"equation\": \"302/2\"}}]</API> => 151</thought> 151 棵美人蕉。这些美人蕉将小道分成了 150 个等长的区间。因此，每两棵美人蕉之间的距离是 $600\\div 150=$<thought>这里需要计算除法，我需要在这里调用 Calculator API<API>[{\"ActionName\": \"Calculator\", \"Args\": {\"equation\": \"600/150\"}}]</API> => 4</thought> 4 米。因此，每两棵美人蕉之间相距4米。\
<reserved_106>{expression}\
<reserved_107>"
                    ),            
                ]),
            ice_token="</E>"
#             template="<reserved_106>一桶奶粉上标有净含量$\\left( 900\\pm 5 \\right)$克，这种奶粉最轻不少于（   ）克．A. $900$B. $905$C. $895$D. $890$\
# <reserved_107>这道题目需要我们理解净含量的概念。净含量$\\left( 900\\pm 5 \\right)$克表示这种奶粉的净含量在$900$克左右，误差范围为$5$克。因此，这种奶粉的净含量最少是$900-5=895$克，最多是$900+5=905$克。题目要求这种奶粉最轻不少于多少克，因此我们需要选择最小的净含量，即$895$克。因此，答案是C。\
# <reserved_106>你能详细给我解答一下 计算：$1\\times 2+2\\times 3+3\\times 4+\\cdots +14\\times 15$ 这个问题的解题步骤吗？\
# <reserved_107>这道题目是一个求和问题，我们可以使用整数拆分法来解决它。原式可以写成$1\\times 2+2\\times 3+3\\times 4+\\cdots +14\\times 15$。我们可以将每一项拆分成两个部分，例如第一项可以拆分成$1\\times 2=2-1$，第二项可以拆分成$2\\times 3=6-2$，以此类推。这样，原式就可以写成$2-1+6-2+12-3+\\cdots +210-14$。我们可以将这些项分组，每组两项，例如第一组是$2-1+6-2=3+4$，第二组是$12-3+20-4=9+16$，以此类推。这样，原式就可以写成$\\left( 3+4 \\right)+\\left( 9+16 \\right)+\\cdots +\\left( 195+196 \\right)+\\left( 210-14 \\right)$。我们可以将每组两项相加，例如第一组是$3+4=7$，第二组是$9+16=25$，以此类推。这样，原式就可以写成$7+25+\\cdots +401+196$。我们可以使用等差数列求和公式来计算这个和，其中首项是$7$，公差是$18$，项数是$14$。因此，原式$=\\frac{14}{2}\\times \\left( 7+196 \\right)=$ <thought>这里需要计算乘法和除法，我需要在这里调用 Calculator API <API>[{\"ActionName\": \"Calculator\", \"Args\": {\"equation\":\"14/2*(7+196)\"}}] </API> => 1120</thought> $1120$。因此，原式的值是1120。\
# <reserved_106>一条乡间小道长$600$米，在这条小道的两侧等距离栽种$302$棵美人蕉（两端要栽），每两棵美人蕉相距多少米？ 这个问题你能帮我解答一下吗\
# <reserved_107>这条小道的长度是 600 米，两侧等距离栽种了 302 棵美人蕉。因为两端也要栽种，所以每侧种了 $302\\div 2=$<thought>这里需要计算除法，我需要在这里调用 Calculator API<API>[{\"ActionName\": \"Calculator\", \"Args\": {\"equation\": \"302/2\"}}]</API> => 151</thought> 151 棵美人蕉。这些美人蕉将小道分成了 150 个等长的区间。因此，每两棵美人蕉之间的距离是 $600\\div 150=$<thought>这里需要计算除法，我需要在这里调用 Calculator API<API>[{\"ActionName\": \"Calculator\", \"Args\": {\"equation\": \"600/150\"}}]</API> => 4</thought> 4 米。因此，每两棵美人蕉之间相距4米。\
# <reserved_106>{expression}\
# <reserved_107>"
            ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer,  max_out_len=2048))

    arith_eval_cfg = dict(
        evaluator=dict(type=ArithEvaluator), 
        pred_role="BOT",
        pred_postprocessor=dict(type=arith_std_first_option_postprocess)
        )

    arith_std_datasets_few.append(
        dict(
            type=ArithDataset,
            abbr=f'arith_std_few_{_name}',
            path=f'./data/arith_std/',
            name=_name,
            reader_cfg=arith_reader_cfg,
            infer_cfg=arith_infer_cfg,
            eval_cfg=arith_eval_cfg)
    )

del _name