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


def compute_line_from_regr(X_data, y_data, regr):
    l_bound = np.min(X_data)
    r_bound = np.max(X_data)
    return [l_bound, r_bound], [l_bound*regr.coef_+regr.intercept_, r_bound*regr.coef_+regr.intercept_]


reg = LinearRegression().fit(x.reshape(-1, 1), y)
plt.scatter(x, y, s=10)
line_x, line_y = compute_line_from_regr(x.reshape(-1, 1), y, reg)
plt.plot(line_x, line_y, color='r')
plt.show()


# this linear regression line shows that it does not fit the data set

'''
Takes raw data in and splits the data into
x_train, y_train, x_test, y_test, x_val,y_val
Returns x_train, y_train, x_test,y_test,x_val,y_val
'''

# adds a column of 1 to windows


def bias_trick(X):
    bias = np.ones((len(X), 1))
    X = np.hstack((bias, X))
    return X


def separate_data(data):
    y = data[:, -1:]
    x = bias_trick[data[:, :-1]]

    return x, y


def train_test_validation_split(data, test_size=.20, validation_size=0.20):
    x, y = separate_data(data)
    split = int(test_size*x.shape[0])
    x_test = x[:split]
    x_val = x[split:(split*2)]
    x_train = x[(split*2):]

    y_test = y[:split]
    y_val = y[split:(split*2)]
    y_train = x[(split*2):]

    print(x_test.shape)
    print(x_val.shape)
    print(x_train.shape)

    return x_train, y_train, y_test, x_val, y_val


'''
Adds columnts to data up to the specified degree. 

EX: if Degree = 3, (x) -> (x,x^2,x^3)
'''


def add_polycols(x, degree):
    x_col = x[:, -1]

    for i in range(2, degree+1):
        x = np.hstack((x, (x_col**i).reshape(-1, 1)))


'''
Takes the target values and predicted values and calculates
the absolute error between them
'''


def mse(y_pred, y_true):
    n = len(y_pred)
    return 0.5*(1/n)*np.sum((y_pred-y_true)**2)


'''
Implementation of the derivative of MSE. 
Returns a vector of the derivations of loss
with respect to each of the dimensions
'''


def mse_derivative(x, y, theta):
    n = len(x)
    x_trans = x.T
    y_hat = np.dot(x, theta)
    return (1/n)*np.dot(x_trans, (y_hat-y))


'''
Computs L@ norm from theta scaled by lambda
Returns a scaler L2 norm. 
'''


def l2norm(theta, lamb):
    return lamb*np.sum(theta*2)


'''
Computes derivative of L2 Norm scaled by lambda
Returns a vector of derivative of L2 norms. 
'''


def l2norm_derivative(theta, lamb):
    return 2*lamb*theta


'''
Computes total cost (cost function + regularization term)
'''


def compute_cost(x, y, theta, lamb):
    y_hat = np.dot(x, theta)
    return mse(y_hat, y)+l2norm(theta, lamb)


'''
Gradient descent step. 
Taxes x, y, theta vector and alpha. 
Returns an updated theta vector
'''


def gradient_descent_step(x, y, theta, alpha, lamb):
    new_mse = mse_derivative(x, y, theta)
    theta = theta - alpha*(new_mse)
    theta + l2norm_derivative(theta, lamb)
    return theta


def polynomial_regression
