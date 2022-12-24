import pytest
import numpy as np

from variable import Variable
from functions import square


def test_forward():
    x = Variable(np.array(2.0))
    y = square(x)
    expected = np.array(4.0)
    assert y.data == expected
