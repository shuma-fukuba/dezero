from core.function import Function


class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy
