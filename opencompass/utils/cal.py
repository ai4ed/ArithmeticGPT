import math
import sympy
from sympy import *
import cmath
from wrapt_timeout_decorator import timeout
import os
# import concurrent.futures

# from utils.logic.logic import SymPyGamma

from opencompass.logic import SymPyGamma
import traceback
import re

from sympy import simplify, latex
from sympy.parsing.latex import parse_latex


def clean_latex(latex_expr):
    # Remove spaces
    latex_expr = latex_expr.replace(" ", "")
    latex_expr = latex_expr.replace("（", "(")
    latex_expr = latex_expr.replace("）", ")")
    latex_expr = latex_expr.replace("÷", "/")
    latex_expr = latex_expr.replace("\sqrt[]", "\sqrt")
    latex_expr = latex_expr.rstrip("=")
    return latex_expr


def convert_decimal_to_fraction(decimal):
    new_result = sp.nsimplify(decimal, rational=True)
    return new_result


def convert_expr(expr):
    return expr.replace(
        lambda x: 1,
        lambda x: convert_decimal_to_fraction(x),
        map=False,
        exact=True,
    )


def latex2exp(latex_str):
    latex_str = clean_latex(latex_str)
    expr = parse_latex(latex_str)
    return expr


def cal_latex(text):
    expr = latex2exp(text)
    result = simplify(expr)
    result = latex(result)
    return str(result)

num_decimal = 6

def evaluate_expression_base(expression):
    # Define some common mathematical functions that can be used by eval()

    expression = expression.replace("^", "**")
    math_functions = {
        "sqrt": cmath.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
        "pow": math.pow,
        "mod": math.fmod,
        "fact": math.factorial,
        "degrees": math.degrees,
        "radians": math.radians,
        "abs": abs,
        "round": round,
        "pi": math.pi,
        "e": math.e
    }

    try:
        result = eval(expression, math_functions)
        return result
    except Exception as e:
        print(f"Uncomputable expression: {expression}")
        print(f"error message: {str(e)}")
        pass



def format_trigonometric(expression: str):
    pattern_trigonometric = re.compile(r"(sin|cos|tan|csc|sec|cot)\((.+?)\)")
    expression_formatted = re.sub(
        pattern_trigonometric,
        lambda x: x.group()
        if "pi" in x.group(2)
        else f"{x.group(1)}(radians({x.group(2)}))",
        expression,
    )
    return expression_formatted


def evaluate_expression_ensembled(args):
    expression = args["equation"]
    if any([op in expression for op in ["sin", "cos", "tan"]]):
        expression = calculate_trigonometric(expression)
    res_prime = evaluate_expression_base(expression)
    if not isinstance(res_prime, complex):
        res = round(res_prime, num_decimal)
    elif "0j" in str(res_prime):
        res = str(round(res_prime.real, num_decimal))
    else:
        if res_prime.real == 0:
            res = str(round(res_prime.imag, num_decimal)) + "i"
        else:
            res = (
                str(round(res_prime.real, num_decimal))
                + (
                    str(round(res_prime.imag, num_decimal))
                    if "-" in str(round(res_prime.imag, num_decimal))
                    else ("+" + str(round(res_prime.imag, num_decimal)))
                )
                + "i"
            )
    return res

# used inside solve_calculator_merged function
def evaluate_expression_ensembled_inner(expression):
    if any([op in expression for op in ["sin", "cos", "tan"]]):
        expression = calculate_trigonometric(expression)
    res_prime = evaluate_expression_base(expression)
    print(res_prime)
    if not isinstance(res_prime, complex):
        res = round(res_prime, num_decimal)
    elif "0j" in str(res_prime):
        res = str(round(res_prime.real, num_decimal))
    else:
        if res_prime.real == 0:
            res = str(round(res_prime.imag, num_decimal)) + "i"
        else:
            res = (
                str(round(res_prime.real, num_decimal))
                + (
                    str(round(res_prime.imag, num_decimal))
                    if "-" in str(round(res_prime.imag, num_decimal))
                    else ("+" + str(round(res_prime.imag, num_decimal)))
                )
                + "i"
            )
    return res


def ceiling(args):
    equation = args["number"]
    result = eval(equation)
    result = math.ceil(result)
    return str(result)


def format_number(number):
    """
    Formatting Numbers functions

    Args:
        number: The number to format

    Returns:
        The formatted number

    """
    if number == int(number):
        number = int(number)  # If the input is an integer, convert it to an integer type
    elif type(number) == float:
        number = round(number, num_decimal)  
        number = str(number).rstrip("0")

    return str(number)


def format_complex_number(complex_num):
    real_part = format_number(complex_num.real)
    imag_part = format_number(complex_num.imag)

    if imag_part == 0:  # If the imaginary part is 0
        return real_part
    elif real_part == 0:  # If the real part is 0
        return str(imag_part) + "i"
    else:  # Both the real and imaginary parts are non-0
        imag_part_str = ("+" + str(imag_part)) if imag_part >= 0 else str(imag_part)
        return str(real_part) + imag_part_str + "i"

def count_trig_functions(expression):
    tri_sum = sum(expression.count(func) for func in ["sin", "cos", "tan"])
    pi_sum = expression.count('pi')
    return tri_sum == pi_sum

def find_matching_parentheses(s):
    stack = []
    pairs = []
    for i, c in enumerate(s):
        if c == '(':
            stack.append(i)
        elif c == ')':
            if stack:
                pairs.append((stack.pop(), i))
            else:
                raise ValueError("No matching left parenthesis for right parenthesis at index {}.".format(i))
    if stack:
        raise ValueError("No matching right parenthesis for left parenthesis at index {}.".format(stack[0]))
    
    record = []
    for p in pairs:
        left = p[0]
        right = p[1]
        parenthese_left = s[left-3: left]
        if parenthese_left in ["sin", "cos", "tan"]:
            parenthese_content = s[left-3: right+1]
            if 'pi' not in parenthese_content:
                re_content = s[left-3: left+1] + 'radians(' + s[left+1: right+1] + ')'
                record.append((parenthese_content, re_content))
    for r in record[::-1]:
        s=s.replace(r[0],r[-1])
    return s

def calculate_trigonometric(expr):
    if 'π' in expr:
        expr = expr.replace('π', 'pi')
    expr = re.sub(r'(?<=[0-9])pi', '*pi', expr)

    if count_trig_functions(expr):
        return expr
    return find_matching_parentheses(expr)

def parenthesis(expr):
    left_parenth = expr.count('(')
    right_parenth = expr.count(')')
    if left_parenth == right_parenth:
        return expr
    elif left_parenth > right_parenth:
        expr = expr + ')'
    else:
        expr = '(' + expr
    return parenthesis(expr)

# @timeout(1)
def solve_calculator_merged(args):
    expression = args["equation"]

    # # Calculate directly with Latex
    if "\\" in expression:
        try:
            result = cal_latex(expression)
            return result
        except:
            pass
    expression = expression.replace("^", "**")
    expression = expression.replace("÷", "/").replace('°', '')
    expression = parenthesis(expression)
    pattern_mod = re.compile(r"(\d+)%(\d+)")
    whether_mod = re.search(pattern_mod, expression)
    try:
        if (
            any(
                [
                    op in expression
                    for op in [
                        "sqrt",
                        "sin",
                        "cos",
                        "tan",
                        "log",
                        "exp",
                        "pow",
                        "fact",
                        "degrees",
                        "radians",
                        "abs",
                        "round",
                        "pi",
                        "mod",
                        "π"
                        "e",
                    ]
                ]
            )
            or whether_mod
        ):
            # Try using a calculator
            try:
                print('expression: ', expression)
                result = evaluate_expression_ensembled_inner(expression)
                res_simplified = nsimplify(result, rational=True)
                if "/" in str(res_simplified):
                    numerator = res_simplified.p
                    denominator = res_simplified.q
                    if len(str(numerator)) > 2 or len(str(denominator)) > 2:
                        res_simplified = round(res_simplified.evalf(), num_decimal)
                return str(res_simplified)
            except Exception as e:
                print(e)
                pass

        # Processing percent
        if expression[-1] == '%':
            expression = expression[:-1] + "/100"
        result = nsimplify(SymPyGamma().eval_input(expression)[-1], rational=True)
        if result.is_complex and not result.is_real:  # if complex, not real
            result = str(result).replace("*I", "i")
        else:
            float_result = format_number(float(result.evalf()))
            if (
                "." in float_result and len(float_result.split(".")[-1]) >= 6
            ):  #if float, more than 6 digits behind the decimal point
                result = str(result)
            else:
                result = float_result
        try:
            float(result)
        except:
            result = evaluate_expression_ensembled_inner(result)
        return result
    except Exception as e:
        print(e)
        print(f"Calculator cannot calculate the expression: {expression}, try other calculation engines")
        if "x" in expression:
            if "=" in expression:  # solution equation
                args = {"unknownVariables": ["x"]}
                args["equations"] = [expression]
                result = solve_equations(args)
            else:
                # simplified equation
                args = {"equation": expression, "unknownVariables": ["x"]}
                result = equation_simplification(args)
            return result
        return ""

@timeout(1)
def solve_calculator(args):
    expression = args["equation"]

    # processing %
    expression = expression.replace("%", "/100")
    expression = expression.replace("^", "**")
    math_functions = {
        "sqrt": cmath.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "exp": math.exp,
        "pow": math.pow,
        "fact": math.factorial,
        "degrees": math.degrees,
        "radians": math.radians,
        "abs": abs,
        "round": round,
        "pi": math.pi,
        "e": math.e,
    }

    try:
        result = nsimplify(SymPyGamma().eval_input(expression)[-1], rational=True)
        if result.is_complex and not result.is_real:  # is complex, not real
            result = str(result).replace("*I", "i")
        else:
            float_result = format_number(float(result.evalf()))
            if (
                "." in float_result and len(float_result.split(".")[-1]) >= 6
            ):  #if float, more than 6 digits behind the decimal point
                result = str(result)
            else:
                result = float_result
        return result
    except Exception as e:
        print(f"Calculator cannot calculate the expression: {expression}, try other calculation engines")
        if "x" in expression:
            if "=" in expression:  # solution equation
                args = {"unknownVariables": ["x"]}
                args["equations"] = [expression]
                result = solve_equations(args)
            else:
                #  simplified equation
                args = {"equation": expression, "unknownVariables": ["x"]}
                result = equation_simplification(args)
            return result
        return ""


def normalize_equation(equation):
    equation = equation.replace(" ", "").replace("%", "/100")
    return equation


def rm_equation_to_left(text):
    left, right = text.split("=")
    return left + f"-({right.strip()})"


def solve_equations(args):
    vs = args["unknownVariables"]
    equations = args["equations"]
    one_equ_list = []
    for one_equ in equations:
        one_equ = rm_equation_to_left(one_equ)
        one_equ_list.append(one_equ)
    input_cmd = f'solve([{",".join(one_equ_list)}],{str(vs)})'
    # solve equ
    result = SymPyGamma().eval_input(input_cmd)[-1]
    if len(result) == 0:
        return ""
    # get result
    v_list = []
    for k, v in result.items():
        if isinstance(v, sympy.Float):
            v = format_number(float(v))
        v_list.append(f"{k}={v}")
    output_str = ",".join(v_list)
    return output_str


def greatest_common_divisor(args):
    numbers = args["number_list"]
    result = numbers[0]
    for num in numbers[1:]:
        result = gcd(result, num)
    return str(result)


def least_common_multiple(args):
    numbers = args["number_list"]
    result = numbers[0]
    for num in numbers[1:]:
        result = lcm(result, num)
    return str(result)


def equation_simplification(args):
    vs = args["unknownVariables"]
    equation = args["equation"]
    if type(equation) == list:
        equation = equation[0]
    equation = normalize_equation(equation)
    if "=" in equation:
        args["equations"] = [equation]
        return solve_equations(args)
    else:
        input_cmd = f"simplify({equation})"
        # print(input_cmd)
        result = SymPyGamma().eval_input(input_cmd)[-1]
        return str(result)


def division(args):
    dividend = eval(args["dividend"])
    divisor = eval(args["divisor"])
    quotient = format_number(dividend // divisor)
    remainder = format_number(dividend % divisor)
    result = f"Quotient is {quotient}, remainder is {remainder}"
    return result


def prime_factor(args):
    number = int(args["number"])
    result = factorint(number)
    return str(result)


def compare_numbers(args):
    a = args["number1"]
    b = args["number2"]
    if a < b:
        output_str = f"{a} is less than {b}"
    elif a > b:
        output_str = f"{a} is greater than {b}"
    else:
        output_str = f"{a} is equal to {b}"
    return output_str


def simplify_rational(args):
    numerator = args["numerator"]
    denominator = args["denominator"]
    x = sympy.Rational(numerator, denominator)
    reduced_fraction = x.simplify()
    return str(reduced_fraction)


def fibonacci_recursive(n):
    if n <= 1:
        return n
    else:
        return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def fibonacci(args):
    n = args["n"]
    result = fibonacci_recursive(int(n))
    return str(result)


# "Calculator": solve_calculator, solve_calculator,
api_map = {
    "Calculator": solve_calculator_merged,
    "SolveEquation": solve_equations,
    "SolveAlgebraicEquation": solve_equations,
    "GreatestCommonDivisor": greatest_common_divisor,
    "LeastCommonMultiple": least_common_multiple,
    "Ceiling": ceiling,
    "Division": division,
    "PrimeFactors": prime_factor,
    "PrimeFactor": prime_factor,
    "CompareNumbers": compare_numbers,
    "SimplifyRational": simplify_rational,
    "AlgebraicEquationSimplification": equation_simplification,
    "Fibonacci": fibonacci,
}


def func_test(api_name, input_text, expected_result):
    global total_test_num
    api = api_map[api_name]
    result = str(api(input_text)).replace(" ", "")

    expected_result = expected_result.replace(" ", "")

    if result != expected_result:
        print(
            f"{api_name} failed. input_text: {input_text}, expected: {expected_result} (dtype={type(expected_result)}), got: {result} (dtype={type(result)})"
        )
    assert result == expected_result
    total_test_num += 1

if __name__ == "__main__":
    solve_calculator_merged({"equation": "cos((52%20)*π)"})
    
