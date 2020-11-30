import tensorflow as tf
from tensorflow import keras


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


fashion_mnist = keras.datasets.fashion_mnist

# splitting training and test data and corresponding labels
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# makes a dictionary for each
class_dict = {i: class_name for i, class_name in enumerate(class_names)}

print(class_dict)


def show_image(index):
    plt.figure()
    plt.imshow(train_images[index], cmap=plt.cm.binary)
    plt.title(class_names[train_labels[index]])
    plt.colorbar()
    plt.show()


show_image(1)
