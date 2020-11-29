from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


# Normalizes the data


def normalize_data(data):
    return (data-np.min(data)/(np.max(data)-np.min(data)))


np.random.seed(33)
x = np.random.uniform(-10, 10, 1000)
poly_coeffs = np.random.uniform(-1, 1, size=(4, 1))
y = poly_coeffs[0]+poly_coeffs[1]*x + poly_coeffs[2] * \
    (x**2)+poly_coeffs[3]*(x**3) + np.random.normal(0, 250, 1000)

x2 = np.random.uniform(-10, 10, 1000)
poly_coeffs = np.random.uniform(-1, 1, size=(4, 1))
y2 = poly_coeffs[0]-2000+poly_coeffs[1]*x2+50 * \
    poly_coeffs[2]*(x2**2)+np.random.normal(0, 250, 1000)
x = np.concatenate([x, x2])
y = np.concatenate([y, y2])
x = normalize_data(x)
y = normalize_data(y)

plt.scatter(x, y, s=10)
plt.show()


poly_data = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
np.random.shuffle(poly_data)
x = poly_data[:, 0]
y = poly_data[:, 1]
