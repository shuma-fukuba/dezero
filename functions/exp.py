import numpy as np

from .function import Function


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs.data
        gx = np.exp(x) * gy
        return gx
