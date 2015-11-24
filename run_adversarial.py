import adversarialnets
import numpy as np
from sklearn.datasets import load_digits

digits = load_digits()
x = digits.data[0:20]
print(x[0].shape)
x /= x.max()
model = adversarialnets.Model(x, 64,20)
model.train()
