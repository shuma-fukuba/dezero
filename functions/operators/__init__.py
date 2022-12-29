from .Mul import Mul
from .Add import Add


def mul(x0, x1):
    return Mul()(x0, x1)


def add(x0, x1):
    f = Add()
    return f(x0, x1)
