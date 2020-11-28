import numpy as np
import matplotlib.pyplot as pyplot


def normalize_data(data):
    return (data-np.min(data)/(np.max(data)-np.min(data)))


np.random.seed(33)
x = np.random.uniform(-10, 10, 1000)
poly_coeffs = np.random.uniform(-1, 1, size=(4, 1))
y = poly_coeffs[0]+poly_coeffs[1]