import numpy as np


class Variable:
    def __init__(self, data) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported.")
        self.data = data
        self.grad = None  # numpyの多次元配列を想定
        self._creator = None

    @property
    def creator(self):
        return self._creator

    @creator.setter
    def creator(self, func):
        self._creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        functions = [self._creator]
        while functions:
            f = functions.pop()
            x, y = f.inputs, f.output
            x.grad = f.backward(y.grad)

            if x._creator is not None:
                functions.append(x._creator)
