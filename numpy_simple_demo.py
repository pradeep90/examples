import numpy as np

def foo() -> None:
    x = np.zeros((300, 400), dtype=np.int)
    y = np.arange(200000).reshape(400, -1)

    z = x @ y
    a = np.zeros((1, 500), dtype=np.int)
    b = z + a
    c = b.T
    d = c.sum(axis=0)
