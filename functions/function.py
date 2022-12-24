from variable import Variable
from utils.as_array import as_array


class Function:
    def __init__(self) -> None:
        self.inputs = None
        self.output = None

    def __call__(self, inputs: Variable):
        x = inputs.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.creator = (self)
        self.inputs = inputs
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()
