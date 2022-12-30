from .variable import Variable
from .function import Function  # noqa: F401
from .operators import add, mul, div, rdiv, rsub, sub, neg, self_pow


def setup_operators():
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = self_pow


setup_operators()
