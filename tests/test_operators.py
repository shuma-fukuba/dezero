import numpy as np
from dezero.core import Variable


def test_add():
    x = Variable(np.array(3.0))
    y = x + x
    expected = np.array(6.0)
    assert y.data == expected


def test_mul():
    x = Variable(np.array(3.0))
    y = x * x
    expected = np.array(9.0)
    assert y.data == expected


def test_sub():
    x = Variable(np.array(3.0))
    y = x - 2.0
    expected = np.array(1.0)
    assert y.data == expected

def test_rsub():
    x = Variable(np.array(3.0))
    y = 2.0 - x
    expected = np.array(-1.0)
    assert y.data == expected


def test_neg():
    x = Variable(np.array(3.0))
    y = -x
    expected = np.array(-3.0)
    assert y.data == expected


def test_div():
    x = Variable(np.array(8.0))
    y = x / 2.0
    expected = np.array(4.0)
    assert y.data == expected


def test_rdiv():
    x = Variable(np.array(2.0))
    y = 8.0 / x
    expected = np.array(4.0)
    assert y.data == expected


def test_pow():
    x = Variable(np.array(3.0))
    c = 4.0
    expected = np.array(81)
    y = x ** c
    assert y.data == expected
