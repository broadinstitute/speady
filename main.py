import numpy as np
import pearson

x = np.random.uniform(size=(500, 20000))
y = np.random.uniform(size=(500, 100))


def add_holes(x, count=100):
    x = x.copy()
    i_v = np.random.choice(range(x.shape[0]), count)
    j_v = np.random.choice(range(x.shape[1]), count)
    for i in range(count):
        x[i_v[i], j_v[i]] = np.NaN
    return x


x_with_nans = add_holes(x, count=40000)
y_with_nans = add_holes(y, count=200)

print(pearson.optim_cy_pearson(x_with_nans, y_with_nans))
