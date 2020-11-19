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


print(lr_dataframe.head(10))
