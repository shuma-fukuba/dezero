from dezero.core.utils import as_array
from .Mul import Mul
from .Add import Add
from .Neg import Neg
from .Sub import Sub
from .Div import Div
from .Pow import Pow


def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1)


def add(x0, x1):
    x1 = as_array(x1)
    f = Add()
    return f(x0, x1)


def neg(x):
    return Neg()(x)


def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)


def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)


def div(x0,  x1):
    x1 = as_array(x1)
    return Div()(x0, x1)


def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)


def self_pow(x, c):
    return Pow(c)(x)
