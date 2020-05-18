import os
import matplotlib.image as mpimg
import numpy as np
from keras.utils import np_utils

from keras.utils import to_categorical
from keras.datasets import mnist
import keras
import matplotlib.pyplot as plt
import Augmentor
from scipy.io import loadmat
import os


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(X_train.shape)

X_train = X_train.reshape(X_train.shape[0], 28, 28 , 1)

X_train=X_train.reshape((-1, 28, 28,1))


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train, 10)

p = Augmentor.Pipeline()

p.rotate(probability=0.3, max_left_rotation=3, max_right_rotation=3)
p.zoom(probability=0.2, min_factor=0.6, max_factor=1.2)
p.random_distortion(probability=0.3, grid_width=4, grid_height=4, magnitude=2)
p.shear(0.3, 0.2, 0.2)
p.status()


for i in range(2):
    g = p.keras_generator_from_array(X_train, Y_train, 9, )
    images, labels = next(g)
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(images[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
        plt.title(np.argmax(labels[i]))
    #show the plot
    plt.show()