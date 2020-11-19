#Jose Lopez
#This code performs linear regression on data from scratch using python. 

#this aims to map feature vectors to a continous value in the range [-infinity, +infinity]


#Information on the data used can be seen in the following link

#http://archive.ics.uci.edu/ml/datasets/Facebook+metrics


#wget http://archive.ics.uci.edu/ml/machine-learning-databases/00368/Facebook_metrics.zip -O ./Facebook_metrics.zip


import wget
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00368/Facebook_metrics.zip'
wget.download(url)

import zipfile
with zipfile.ZipFile('./Facebook_metrics.zip', 'r') as zip_ref:
    zip_ref.extractall('./')
