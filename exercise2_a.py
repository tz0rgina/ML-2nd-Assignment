# Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import to_categorical
from keras.datasets import mnist
import keras
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split


SVHN_directory = os.path.join(os.path.join(os.getcwd(), os.path.join("svhn", "train.mat")))
# load .mat file
data_raw = loadmat(SVHN_directory)
data = np.array(data_raw['X'])
print(data.shape)

# make correct shape
data = np.moveaxis(data, -1, 0)

plt.show()

labels = data_raw['y']

# Preprocess input data

data = data.astype('float32')

data /= 255

# Preprocess class labels
labels[labels == 10] = 0
labels = to_categorical(labels.reshape([-1, 1]))

X_train, X_valid, Y_train, Y_valid = train_test_split(data, labels, test_size=0.10, shuffle= True)
 
# Define model architecture
model = Sequential()

model.add(Convolution2D(9, (3, 3), padding='same', activation='relu', input_shape=(32,32,3)))
model.add(MaxPooling2D(pool_size=(3,3)))

# Adding a second convolutional layer
model.add(Convolution2D(36, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

# Adding a third convolutional layer
model.add(Convolution2D(49, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))
 
 #Adding a flatten layer
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

model.summary()

# Compile model categorical_crossentropy
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

early_stop=keras.callbacks.EarlyStopping(monitor='val_loss',
                              verbose=1, patience=5, mode='min') #stop training when val_loss begins increase
						  
# Fit model on training data
history2=model.fit(X_train, Y_train, 
          batch_size=64, epochs=300, verbose=1, validation_data=(X_valid, Y_valid), callbacks=[early_stop])

model.save('model_2a.h5') #saving the model
import pickle

with open('trainHistory', 'wb') as handle: # saving the history of the model
   pickle.dump(history2.history, handle)
