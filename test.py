import numpy as np
from variable import Variable
from functions import add

x = Variable(np.array(3.0))
y = add(x, x)
y.backward()
print(x.grad)

x.clean_grad()
y = add(add(x, x), x)
y.backward()
print(x.grad)
