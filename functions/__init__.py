from .exp import Exp
from .square import Square
from .add import Add

def exp(x):
    f = Exp()
    return f(x)


def square(x):
    f = Square()
    return f(x)


def add(x0, x1):
    f = Add()
    return f(x0, x1)
