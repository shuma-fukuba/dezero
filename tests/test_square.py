import pytest
import numpy as np

from variable import Variable
from functions import square
from utils.numeric_diff import numeric_diff


def test_forward():
    x = Variable(np.array(2.0))
    y = square(x)
    expected = np.array(4.0)
    assert y.data == expected


def test_backward():
    x = Variable(np.array(3.0))
    y = square(x)
    y.backward()
    expected = np.array(6.0)
    assert x.grad == expected


def test_gradient_check():
    x = Variable(np.random.rand(1))
    y = square(x)
    y.backward()
    num_grad = numeric_diff(square, x)
    flag = np.allclose(x.grad, num_grad)
    assert flag == True
