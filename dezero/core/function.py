import weakref
from dezero.core import Variable
from dezero.core.utils import as_array, as_variable
from dezero.core.config import Config


class Function:
    def __init__(self) -> None:
        self.inputs = None
        self.outputs = None
        self.generation = None

    def __call__(self, *inputs: Variable):
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.creator = self
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError()
