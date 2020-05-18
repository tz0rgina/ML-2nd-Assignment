# Import libraries and modules
import numpy as np
np.random.seed(123)  # for reproducibility

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
from pickle import load
import matplotlib.pyplot as plt
import pandas as pd

# Recreate the exact same model, including its weights and the optimizer from exercise2_a
model = keras.models.load_model('model_2a.h5')

with open('trainHistory_2a', 'rb') as handle: # loading old history 
    oldhstry = load(handle)

# Plotting the Accuracy vs Epoch Graph
plt.plot(oldhstry['accuracy'])
plt.plot(oldhstry['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Plotting the Loss vs Epoch Graphs
plt.plot(oldhstry['loss'])
plt.plot(oldhstry['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# Show the model architecture
model.summary()

SVHN_directory = os.path.join(os.path.join(os.getcwd(), os.path.join("svhn", "test.mat")))

# load .mat file
data_raw = loadmat(SVHN_directory)
data = np.array(data_raw['X'])
print(data.shape)

# make correct shape
data = np.moveaxis(data, -1, 0)
print(data.shape)


labels = data_raw['y']

# Preprocess input data

X_test = data.astype('float32')

X_test /= 255

# Preprocess class labels
labels[labels == 10] = 0
Y_test = np_utils.to_categorical(labels.reshape([-1, 1])) # need of understanding how reshape([-1, 1]) works


# Evaluate model on test data
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

pred = model.predict(np.array(X_test))

actual_results = []
predicted_results = []
for i in range(len(labels)):

    actual_results.append(np.argmax(Y_test)[i])
    predicted_results.append(np.argmax(pred[i]))
	
actual_results=pd.Series(actual_results, name="Actual")	
predicted_results=pd.Series(predicted_results, name="Predicted")	

	
print(pd.crosstab(actual_results,predicted_results))


