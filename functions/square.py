from .function import Function


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        # gy: 出力から伝わる微分
        x = self.inputs.data
        gx = 2 * x * gy
        return gx
