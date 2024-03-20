import sympy as sp
import numpy as np
from easier_excel.Colorful_Console import ColoredText as CT


def math_integral(x=None, fx=None, a=None, b=None):
    """
    [函数功能] 求解定积分：\int_{a}^{b} f(x)dx
    [注意事项] 请务必传入Symbol类型的变量
    [使用方法]：
        x_sym = sp.symbols('x')
        a_sym, b_sym = sp.symbols('a b')
        expression = x_sym ** 2 + 2 * x_sym + sp.sin(x_sym)
        result = xm_mf.math_integral(x_sym, expression, a_sym, b_sym)
        print(result)
    :param x: 微分元 differential
    :param fx: 被积函数 integrand
    :param a: 积分下限
    :param b: 积分上限
    :return: Symbol类型的表达式
    """
    if not isinstance(x, sp.Symbol) or not isinstance(fx, sp.Basic):
        print(CT("ERROR in").red(), CT("math_integral").yellow(), CT("请传入有效的x, f(x)的表达式").red())
        exit(114)
    else:
        if not isinstance(a, (sp.Symbol, int, float)) or not isinstance(b, (sp.Symbol, int, float)):
            print(CT("Tip in").pink(), CT("math_integral").yellow(),
                  CT("因为没有传入有效的积分上下限，这里按照不定积分计算").pink())
            integral = sp.integrate(fx, x)
            return integral
        else:
            integral = sp.integrate(fx, (x, a, b))
            return integral

def math_solve(x=None, fx=None, symbol="=", b=0):
    """
    [函数功能] 求解等式/不等式，如 f(x)=b
    [注意事项] 请务必传入Symbol类型的变量
    [使用方法]：
        x_sym = sp.symbols('x')
        expression = x_sym ** 2 + 2 * x_sym
        result = xm_mf.math_solve(x_sym, expression, symbol='<', b=0)
        print(result)
    :param x:
    :param fx:
    :param symbol:
    :param b:
    :return:
    """
    if not isinstance(x, sp.Symbol) or not isinstance(fx, sp.Basic):
        print(CT("ERROR in").red(), CT("math_solve").yellow(), CT("请传入有效的x, f(x)的表达式").red())
        exit(114)
    else:
        if symbol == "=" or symbol == "==":
            result = sp.solve(sp.Eq(fx, b), x)
        elif symbol == "<":
            result = sp.solveset(fx - b < 0, x, domain=sp.S.Reals)
        elif symbol == ">":
            result = sp.solveset(fx - b > 0, x, domain=sp.S.Reals)
        elif symbol == "<=":
            result = sp.solveset(fx - b <= 0, x, domain=sp.S.Reals)
        elif symbol == ">=":
            result = sp.solveset(fx - b >= 0, x, domain=sp.S.Reals)
        else:
            print(CT("ERROR in").red(), CT("math_solve").yellow(), CT("请检查你的符号是否输入正确").red())
            exit(114)
        return result


# todo 求导
