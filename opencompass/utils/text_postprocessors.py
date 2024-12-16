import re

from opencompass.registry import TEXT_POSTPROCESSORS


@TEXT_POSTPROCESSORS.register_module('general')
def general_postprocess(text: str) -> str:
    # Cut off the first newline, period, or comma
    truncated_text = re.split(r'[\n.,]', text, 1)[0]

    # Remove punctuation
    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    # Remove article
    no_articles = re.sub(r'\b(a|an|the)\b','',no_punctuation,flags=re.IGNORECASE)

    # Remove duplicated blank spaces
    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()

    return cleaned_text


@TEXT_POSTPROCESSORS.register_module('general_cn')
def general_cn_postprocess(text: str) -> str:
    truncated_text = re.split(r'[\n.,]', text, 1)[0]

    no_punctuation = re.sub(r'[^\w\s]', '', truncated_text)

    no_articles = re.sub(r'\b(a|an|the)\b','',no_punctuation,flags=re.IGNORECASE)

    cleaned_text = re.sub(r'\s+', ' ', no_articles).strip()
    import jieba
    cleaned_text = ' '.join(jieba.cut(text))
    return cleaned_text


@TEXT_POSTPROCESSORS.register_module('first-capital')
def first_capital_postprocess(text: str) -> str:
    for t in text:
        if t.isupper():
            return t
    return ''

def get_fenshu4ape210k(mixed_number):
    if not mixed_number:
        return mixed_number
    if '(' not in mixed_number or '/' not in mixed_number:
        return mixed_number
    if mixed_number.count('(')!=1 or mixed_number.count('/')!=1:
        return mixed_number
    integer_part, fraction_part = mixed_number.split('(')
    fraction_part = fraction_part.rstrip(')')
    if not integer_part:
        return fraction_part
    else:
        numbers = fraction_part.split('/')
        whole_number = int(integer_part)
        numerator = int(numbers[0])
        denominator = int(numbers[1])
        full_fraction = whole_number * denominator + numerator
        return f'{full_fraction}/{denominator}'


def get_fenshu(str_ans):
    numbers_re= re.findall(r'(\d+)\(\((\d+)\)\/\((\d+)\)\)', str_ans)
    if numbers_re:
        numbers =numbers_re[-1]
        whole_number = int(numbers[0])
        numerator = int(numbers[1])
        denominator = int(numbers[2])
        improper_fraction_numerator = whole_number * denominator + numerator
        return True,f'{improper_fraction_numerator}/{denominator}'
    else:
        return False,str_ans

@TEXT_POSTPROCESSORS.register_module('merge_dataset')
def merge_first_option_postprocess(text: str, options: str) -> str:
    """Find first valid option for text."""

    patterns = [
         f'answer is ({options})',
        f'[Tt]he correct answer is ({options})',
        f'答案是.*?({options})',
        f'答案:.*?({options})',
        f'答案为.*?({options})',
        f'故选.*?({options})',
        f'答案应该是.*?({options})',
        f'答案.*\\n*=({options})',
        f'答案：.*?({options})',
        f'答案.*?\\n*\$*({options})', #fix bbh for bai
        f'answer is.*?\\n*\$*({options})',
        f'answer:.*?\\n*\$*({options})',
        f'Answer:.*?\\n*\$*({options})',
        f'Answer.*?\\n*\$*({options})',
        f'所以.*?=({options})',
        f'所以.*?结果.*?({options})',
        f'所以.*?结果.*?({options})',
        f'所以.*?({options})',
        f'答：.*?({options})',
        f'答:.*?({options})',
        f'[sS]o,.*?=\s*\\n*({options})',
        f'####.*?({options})',
        f'[tT]herefore,.*?=\s*\\n*({options})',
        f'因此.*?=({options})',
        f'因此.*?({options})',
        f'=\s?({options})',
        f'({options})',
    ]

    regexes = [re.compile(pattern) for pattern in patterns]
    for regex in regexes:
        match = regex.findall(text)
        if match:
            if 'frac{' in match[0]:
                # LaTeX
                # Use regular expressions to extract the numerator and denominator
                match2 = re.search(r'frac\{(-?\d+)\}\{(\d+)\}', match[0])
                if match2:
                    numerator, denominator = map(int, match2.groups())
                    outputs = f'{numerator}/{denominator}'
                else:
                    outputs = match[0]
            elif '[sS]o,' in str(regex):
                outputs = match[-1]
            else:
                outputs = match[0]
            print(f"---{outputs}--",regex,text)
            return outputs
    print("---{no ans}--",text)
    return ''




@TEXT_POSTPROCESSORS.register_module('merge_math23k')
def merge_first_option_postprocess4math23k(text: str, options: str) -> str:
    """Find first valid option for text."""

    patterns = [
        f'answer is ({options})',
        f'[Tt]he correct answer is ({options})',
        f'答案是.*?({options})',
        f'答案:.*?({options})',
        f'答案为.*?({options})',
        f'故选.*?({options})',
        f'答案应该是.*?({options})',
        f'答案.*\\n*=({options})',
        f'答案：.*?({options})',
        f'答案.*?\\n*\$*({options})', #fix bbh for bai
        f'answer is.*?\\n*\$*({options})',
        f'answer:.*?\\n*\$*({options})',
        f'Answer:.*?\\n*\$*({options})',
        f'Answer.*?\\n*\$*({options})',
        f'所以.*?=({options})',
        f'所以.*?结果.*?({options})',
        f'所以.*?结果.*?({options})',
        f'所以.*?({options})',
        f'答：.*?({options})',
        f'答:.*?({options})',
        f'####.*?({options})',
        f'因此.*?=({options})',
        f'因此.*?({options})',
        f'=\s?({options})',
        f'({options})',
    ]
    preifx_text = text[:15]
    pre_pattern = [
        f'.*\\n*=\s?({options})',
        f'({options})',
    ]
    
    pre_regexes = [re.compile(pattern) for pattern in pre_pattern]
    for pre_re in pre_regexes:
        match = pre_re.findall(preifx_text)
        if match:
            return match[0]
    regexes = [re.compile(pattern) for pattern in patterns]
    for regex in regexes:
        match = regex.findall(text)
        if match:
            if 'frac{' in match[0]:
                # Convert to fraction
                # LaTeX expression
                # Use regular expressions to extract the numerator and denominator
                match2 = re.search(r'frac\{(-?\d+)\}\{(\d+)\}', match[0])
                if match2:
                    numerator, denominator = map(int, match2.groups())
                    outputs = f'{numerator}/{denominator}'
                else:
                    outputs = match[0]
            elif '=' in str(regex):
                outputs = match[-1]
            else:
                outputs = match[0]
            # print(f"---{outputs}--",regex,text)
            return outputs
    print("---{no ans}--",text)
    return ''


@TEXT_POSTPROCESSORS.register_module('merge_all')
def merge_first_option_postprocess4all(text: str, options: str) -> str:
    """Find first valid option for text."""
    text = text.strip()
    text = text.replace('，', ',').replace('\\', '')
    pattern = r"(\d+(?:\.\d+)?)%"
    
    def replace_percentage(match):
        number = float(match.group(1))
        return str(number / 100)

    text = re.sub(pattern, replace_percentage, text)
    
    patterns = [
        f'answer is ({options})',
        f'[Tt]he correct answer is ({options})',
        f'答案是.*?({options})',
        f'答案:.*?({options})',
        f'答案为.*?({options})',
        f'故选.*?({options})',
        f'答案应该是.*?({options})',
        f'答案.*\\n*=({options})',
        f'答案：.*?({options})',
        f'答案.*?\\n*\$*({options})', #fix bbh for bai
        f'answer is.*?\\n*\$*({options})',
        f'answer:.*?\\n*\$*({options})',
        f'Answer:.*?\\n*\$*({options})',
        f'Answer.*?\\n*\$*({options})',
        f'所以.*?=({options})',
        f'所以.*?结果.*?({options})',
        f'所以.*?结果.*?({options})',
        f'所以.*?({options})',
        f'答：.*?({options})',
        f'答:.*?({options})',
        f'####.*?({options})',
        f'[sS]o,.*?=\s*\\n*({options})',
        f'[tT]herefore,.*?=\s*\\n*({options})',
        f'因此.*?=({options})',
        f'因此.*?({options})',
        f'=\s?({options})',
        f'({options})',
    ]
    preifx_text = text[:15]
    pre_pattern = [
        f'\n\s?({options}).*?\s?\n',
        f'=({options}).*?\s?\n',
        f'({options}).*?\s?\n'
    ]
    
    pre_regexes = [re.compile(pattern) for pattern in pre_pattern]
    for pre_re in pre_regexes:
        match = pre_re.findall(preifx_text)
        if match:
            return match[0]
    regexes = [re.compile(pattern) for pattern in patterns]
    for regex in regexes:
        match = regex.findall(text)
        if match:
            if 'frac{' in match[0]:
                match2 = re.search(r'frac\{(-?\d+)\}\{(\d+)\}', match[0])
                if match2:
                    numerator, denominator = map(int, match2.groups())
                    outputs = f'{numerator}/{denominator}'
                else:
                    outputs = match[0]
            elif '=' in str(regex) or 'sS]o' in str(regex):
                outputs = match[-1]
            else:
                outputs = match[0]
            print(f"==={outputs}===",regex,text)
            return outputs
    print("==={no ans}===",text)
    return ''

def first_option_postprocess(text: str, options: str) -> str:
    """Find first valid option for text."""

    patterns = [
        f'[Tt]he answer is ([{options}])',
        f'[Tt]he correct answer is ([{options}])',
        f'答案是.*?\\n*([{options}])',
        f'答案为.*?\\n*([{options}])',
        f'固选.*?([{options}])',
        f'故选.*?([{options}])',
        f'answer is.*?\\n*([{options}])',
        f'答案应该是.*?([{options}])',
        f"答案是?\s?([{options}])",
        f"答案是?\s?：([{options}])",
        f"答案是?\s?:([{options}])",
        f"答案应该?是\s?([{options}])",
        f"答案应该?选\s?([{options}])",
        f"答案为\s?([{options}])",
        f"选择\s?\\n*([{options}])",
        f'答案.*?\\n*\$*([{options}])', #fix bbh for bai
        f'answer.*?\\n*([{options}])',
        f'Answer.*?\\n*([{options}])',
        f'选.*?([{options}])',
        f'####.*?([{options}])',
        f'因此.*?([{options}])',
        f'([{options}])',
    ]

    regexes = [re.compile(pattern) for pattern in patterns]
    for regex in regexes:
        match = regex.findall(text)
        if match:
            outputs = match[0]
            for i in options:
                if i in outputs:
                    print(f"---{i}--",regex,text)
                    return i
    print('==null==',text)
    return ''


def option_extract(text: str) -> str:
    text = text.strip()
    text = text.replace('A, B, C or D', '').replace('Answer', 'answer').replace(": ", ":").replace("：", ":").replace(":\n", ':')
    text = re.sub(r"\([^()]*\)", "", text)
    if len(text) == 0:
        return text
    if text[0] in ['A', 'B', 'C', 'D']:
        return text[0]
    anstr1= ['answer is', '答']
    temp1 = 0
    for s in anstr1:
        if s in text:
            matches = re.findall(r"(?:{})(.*)".format(s), text)
            if len(matches) != 0:
                temp1 = 1
                break
    if temp1 == 0 :
        anstr2= ['so ', 'Therefore', 'therefore', '因此', '所以', '故']
        matches = re.findall(r'(.*)', text)
        matches = list(filter(None, matches))
        mm = ''
        temp2 = 0
        for m in reversed(matches) :
            for s2 in anstr2:
                if s2 in m:
                    matches = m
                    if len(matches) != 0:
                        temp2 = 1
                        break
            if temp2 == 1:
                break
    if len(matches) == 0:
        matches =  text

    for s in str(matches):
        if s in ['A', 'B', 'C', 'D']:
            text = s
            break
    if len(text) == 0:
        text = ''
    if text[0] not in ['A', 'B', 'C', 'D']:
        text = ''
    return text

def answer_postprocess(text: str) -> str:
    text = text.strip()
    text = text.replace('，', ',').replace(', ', ',').replace('\n\n','\n').replace('：', ':').replace(':\n',':').replace('=\n','=')
    text = text.replace('是\n', '是').replace('为\n', '为')
    pattern = r'(-?\d+\.?\d*)/(-?\d+\.?\d*)'
    # Find the matching division operator and evaluate it
    def evaluate_fraction(match):
        numerator = float(match.group(1))
        denominator = float(match.group(2))

        if denominator != 0:
            result = numerator / denominator
            return str(result)
        # return match

    text = re.sub(pattern, evaluate_fraction, text)
    text = re.sub(r"\([^()]*\)", "", text)
    matches = ''
    anstr1= ['answer is', '答', 'answers are', 'numbers are', '####']

    temp1 = 0
    for s in anstr1:
        if s in text:
            matches = re.findall(r"(?:{})(.*)".format(s), text)[0]
            if bool(re.search(r'\d', str(matches))) == True:
                break
    temp2 = 0
    if len(matches) == 0 or bool(re.search(r'\d', str(matches))) == False:
        anstr2= ['so ', 'So ', 'So,', 'Therefore', 'therefore', '因此', '所以', '=>']
        match = re.findall(r'(.*)', text)
        match = list(filter(None, match))

        for m in reversed(match) :
            for s2 in anstr2:
                if s2 in m:
                    matches = re.findall(r"(?:{})(.*)".format(s2), m)
                    if bool(re.search(r'\d', str(matches))) == True:
                        temp2 = 1
                        break
            if temp2 == 1:
                break    
    if len(matches) == 0 or bool(re.search(r'\d', str(matches))) == False:
        anstr3= ['Step-by-step', 'step-by-step', 'Explanation', 'explanation', 'Comment']
        for s3 in anstr3:
            if s3 in text:
                matches = text[ : text.find(s3)]
                if bool(re.search(r'\d', str(matches))) == True:
                    break
                
    anstr4= ['A:','Answer:','$']
    if len(matches) == 0 or bool(re.search(r'\d', str(matches))) == False:
        for s4 in anstr4:
            if s4 in text:
                matches = re.findall(r"(?:{})(.*)".format(s4), text)[0]
                if bool(re.search(r'\d', str(matches))) == True:
                    break
    
    if len(matches) == 0 or bool(re.search(r'\d', str(matches))) == False:
        matches = text.split('\n')[-1]
        
    print("mateches is:", matches)
    if len(matches) == 0 or bool(re.search(r'\d', str(matches))) == False:
        matches =  text
    return matches


def one_answer_extract(text: str) -> str:
    matches = answer_postprocess(text)
    numbers = re.findall(r'[-+]?\d+(?:,\d+)?(?:\.\d+)?', str(matches))
    if len(numbers) == 0:
        return text
    text = numbers[-1].strip().strip('.,?!\"\';:')
    text = text.replace(",", '')
    text = round(float(text), 2)
    text = str(text)
    return text
    
def multi_answer_extract(text: str) -> str:
    matches = answer_postprocess(text)
    value = re.findall(r'[-+]?\d+(?:,\d+)?(?:\.\d+)?', str(matches))
    print("value : ", value)
    text = []
    if value:
        for v in value:
            if ',' in v:
                for vs in v.split(','):
                    text.append(round(float(vs), 2))

            v = v.replace(',', '')
            text.append(round(float(v), 2))
    return text


def math_answer_extract(text: str) -> str:
    text = text.strip()
    text = text.replace('，', ',').replace(', ', ',').replace('\n\n','\n').replace('：', ':').replace(':\n',':').replace('=\n','=')
    text = text.replace('是\n', '是').replace('为\n', '为')
    pattern = r'(-?\d+\.?\d*)/(-?\d+\.?\d*)'
    print("1111text is ", text)
    def evaluate_fraction(match):
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        if denominator != 0:
            result = numerator / denominator
            return str(result)

    text = re.sub(pattern, evaluate_fraction, text)
    text = re.sub(r"\([^()]*\)", "", text)
    print("222text is ", text)
    matches = ''
    anstr1= ['answer is', '答', '####', 'so ', 'So ', 'So,', 'Therefore', 'therefore', '因此', '所以']
    temp1 = 0
    for s in anstr1:
        if s in text:
            matches = re.findall(r"(?:{})(.*)".format(s), text)[-1]
            if bool(re.search(r'\d', str(matches))) == True:
                break
    if len(matches) != 0 and bool(re.search(r'\d', str(matches))) == True:
        numbers = re.findall(r'[-+]?\d+(?:,\d+)?(?:\.\d+)?', str(matches))

        text = numbers[-1].strip().strip('.,?!\"\';:')
        text = text.replace(",", '')
        text = str(text)
        print("text: ", text)
        return text
    else:
        matches = re.findall(r"=> ([\d\./+-]*)", text)
        if len(matches) != 0 and bool(re.search(r'\d', str(matches))) == True:
            outputs = matches[-1] 
            if outputs.endswith('.'):
                outputs = outputs+'0'
            print('=> :', outputs)
            return outputs
        else:
            matches = re.findall(r"=>([\d\./+-]*)", text)
            if len(matches) != 0 and bool(re.search(r'\d', str(matches))) == True:
                numbers = re.findall(r'[-+]?\d+(?:,\d+)?(?:\.\d+)?', str(text))
                outputs = matches[-1] 
                if outputs.endswith('.'):
                    outputs = outputs+'0'
                print('=>:', outputs)
                return outputs
    numbers = re.findall(r'[-+]?\d+(?:,\d+)?(?:\.\d+)?', str(text))
    if len(numbers) != 0:
        print('numbers[-1]: ', numbers[-1])
        return numbers[-1]
    return ''


def last_answer_postprocess(text: str) -> str:
    text = text.strip()
    pattern = r'(-?\d+\.?\d*)/(-?\d+\.?\d*)'
    # Find the matching division operator and perform the calculation
    def evaluate_fraction(match):
        numerator = float(match.group(1))
        denominator = float(match.group(2))
        if denominator != 0:
            result = numerator / denominator
            return str(result)
    text = re.sub(pattern, evaluate_fraction, text)
    matches = re.sub(r'\(.*?\)', '', str(text))
    numbers = re.findall(r'[-+]?\d+(?:,\d+)?(?:\.\d+)?', str(matches))
    if len(numbers) != 0:
        print('numbers: ', numbers[-1])
        return numbers[-1]
    return ''

@TEXT_POSTPROCESSORS.register_module('first-capital-multi')
def first_capital_postprocess_multi(text: str) -> str:
    match = re.findall(r'([A-F]+)', text)
    if match:
        return match[-1]
    return ''


def last_option_postprocess(text: str, options: str) -> str:
    match = re.findall(rf'([{options}])', text)
    if match:
        return match[-1]
    return ''
