import numpy as np


class Variable:
    def __init__(self, data) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)} is not supported.")
        self.data = data
        self.grad = None
        self._creator = None
        self.generation = 0

    @property
    def creator(self):
        return self._creator

    @creator.setter
    def creator(self, func):
        self._creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        functions = []
        seen = set()

        def add_func(f):
            if f not in seen:
                functions.append(f)
                seen.add(f)
                functions.sort(key=lambda x: x.generation)

        add_func(self._creator)

        while functions:
            f = functions.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            # 微分の記憶をリセット
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # yはweakref

    def clean_grad(self):
        self.grad = None
