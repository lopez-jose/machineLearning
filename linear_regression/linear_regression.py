# Jose Lopez
# This code performs linear regression on data from scratch using python.

# this aims to map feature vectors to a continous value in the range [-infinity, +infinity]


# Information on the data used can be seen in the following link

# http://archive.ics.uci.edu/ml/datasets/Facebook+metrics


# wget http://archive.ics.uci.edu/ml/machine-learning-databases/00368/Facebook_metrics.zip -O ./Facebook_metrics.zip

import numpy as np
import pandas as pd
import zipfile
import os
import wget
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00368/Facebook_metrics.zip'
PATH = './Facebook_metrics.zip'
if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
    print("File exists and is readable")
else:
    wget.download(url)

with zipfile.ZipFile('./Facebook_metrics.zip', 'r') as zip_ref:
    zip_ref.extractall('./')


np.random.seed(144)


def shuffle_data(data):
    np.random.shuffle(data)


lr_dataframe = pd.read_csv('dataset_Facebook.csv', sep=';')
print(lr_dataframe.head(10))


lr_dataframe.dropna(inplace=True)


columns_to_drop = ['Type', 'Lifetime Post Total Reach', 'Lifetime Post Total Impressions',
                   'Lifetime Engaged Users', 'Lifetime Post Consumers',
                   'Lifetime Post Consumptions',
                   'Lifetime Post Impressions by people who have liked your Page',
                   'Lifetime Post reach by people who like your Page',
                   'Lifetime People who have liked your Page and engaged with your post',
                   'comment', 'like', 'share']

lr_dataframe.drop(columns=columns_to_drop, inplace=True)


def normalize_col(col):
    return (col-col.min())/(col.max()-col.min())


lr_dataframe = lr_dataframe.apply(normalize_col)

# Get Entries as a numpy array

lr_data = lr_dataframe.values[:, :]

shuffle_data(lr_data)


print(lr_dataframe.head(10))


'''
Combines one column of all ones and the matrix X to account for the bias term
(setting x_0 = 1) - [Hint: you may want to use np.hstack()]
Takes input matrix X
Returns the augmented input
'''

'''
hello
'''
# Adds a column of 1 to dataset for the bias term


def bias_trick(X):
    bias = np.ones((len(X), 1))
    X = np.hstack((bias, X))
    return X


'''
Separates feature vectors and targets
Takes raw data and returns X as the matrix of feature vector and Y as the Vector
of targets
'''


def separate_data(data):
    # here Y is the data from the beginning to -1 column from the end
    y = data[:, -1:]
    x = bias_trick(data[:, :-1])
    print(y.shape)
    print(x.shape)
    return x, y

    '''
    Takes raw data in and splits the data into
    X_train, y_train, X_test, y_test

    Returns X_train,y_train,X_test,y_test
    '''


def train_test_split(data, train_size=.80):
    x, y = separate_data(data)
    per = int(train_size*x.shape[0])
    x_train = x[:per]
    x_test = x[per:]
    y_train = y[:per]
    y_test = y[per:]

    return x_train, y_train, x_test, y_test


train_test_split(lr_data)


# Now we begin to train and test the model.

'''
Takes the target values and predicted values and calculates the squared
error between them
'''


def mse(y_pred, y_true):
    n = len(y_pred)
    return 0.5(1/n)*np.sum((y_pred-y_true)**2)


def mse_derivative(x, y, theta):
    n = len(x)
    x_trans = x.T  # this makes the transpose of X
    y_hat = np.dot(x, theta)
    return(1/n)*np.dot(x_trans)


'''
Gradient descent step.

Take x,y, theta and alpha
and returns an updated theta vector.
'''


def gradient_descent_step(x, y, theta, alpha):
    new_mse = mse_derivative(x, y, theta)
    theta = theta-alpha(new_mse)
    return theta


def linear_regression(data, num_epochs=30000, alpha=0.00005):
    x_train, y_train, x_test, y_test = train_test_split(data)

    train_errors = []
    test_errors = []

    # Defining theta

    theta = np.zeros((x_train.shape[1], 1))

    for i in range(num_epochs):
        train_error = mse(np.dot(x_train, theta), y_train)

        # adds current error to train_errors
        train_errors.append(train_error)

        test_error = mse(np.dot(x_test, theta), y_test)
        test_errors.append(test_error)

        # Run gradient descent on the training set
        theta = gradient_descent_step(x_train, y_train, theta, alpha)
    return theta, train_errors, test_errors
