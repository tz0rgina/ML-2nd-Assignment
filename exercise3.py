import os
import matplotlib.image as mpimg
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import *
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import to_categorical
from keras.datasets import mnist
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import Augmentor
from scipy.io import loadmat
import os
from keras.preprocessing.image import ImageDataGenerator


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

print(X_train.shape)

X_train = X_train.reshape(X_train.shape[0], 28, 28 , 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28 , 1)
X_train=X_train.reshape((-1, 28, 28,1))
X_test=X_test.reshape((-1, 28, 28,1))
print(X_train.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train, 10)
print(Y_train.shape)
Y_test = np_utils.to_categorical(Y_test, 10)


p = Augmentor.Pipeline()
p.rotate(probability=0.3, max_left_rotation=3, max_right_rotation=3)
p.zoom(probability=0.2, min_factor=0.6, max_factor=1.2)
p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=4)
p.shear(0.3, 0.2, 0.2)
p.status()

for i in range(2):
    g = p.keras_generator_from_array(X_train, Y_train, 9)
    images, labels = next(g)
    for i in range(0, 9):
        plt.subplot(330 + 1 + i)
        plt.imshow(images[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
        plt.title(np.argmax(labels[i]))
    #show the plot
    plt.show()
    
val_p = Augmentor.Pipeline()
val_p.status()

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    zca_whitening=True,
    brightness_range = (0.5, 4.0),
    zca_epsilon=1e-1,
    )
    
#datagen.fit(X_train)

"""
# Define model architecture

model = Sequential()
model.add(Convolution2D(25, (5, 5), padding='same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
#model.add(Dropout(0.4))
model.add(Convolution2D(50, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
#model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))
"""

# Define model architecture i phone app

model = Sequential()
model.add(Convolution2D(32, (3, 3), padding='same', activation='relu', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=2))
#model.add(Dropout(0.4))
model.add(Convolution2D(32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(10, activation='softmax'))


# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.summary()
"""
batch_size=64
epochs=300 
early_stop=keras.callbacks.EarlyStopping(monitor='val_accuracy',
                              verbose=1, patience=5, mode='max') #stop training when val_loss begins increase

X_batch, y_batch = next(datagen.flow(X_train,Y_train, batch_size=len(X_train)))
print(X_batch.shape)
history3_12 = model.fit_generator(p.keras_generator_from_array(X_batch, y_batch, batch_size=batch_size),
        epochs = epochs, steps_per_epoch = X_train.shape[0]//batch_size, 
        validation_data=val_p.keras_generator_from_array(X_test,Y_test, batch_size=batch_size), validation_steps=(X_test.shape[0])//batch_size,callbacks=[early_stop], verbose=1)

history3drop = model.fit_generator(p.keras_generator_from_array(X_train,Y_train, batch_size=batch_size),
        epochs = epochs, steps_per_epoch = X_train.shape[0]//batch_size, 
        validation_data=val_p.keras_generator_from_array(X_test,Y_test, batch_size=batch_size), validation_steps=(X_test.shape[0])//batch_size,callbacks=[early_stop], verbose=1) #*used for history3_4*
"""

history3app=model.fit(X_train, Y_train, batch_size=512, epochs=6, verbose=1, validation_data=(X_test, Y_test))
# Evaluate model on test data
score = model.evaluate(X_test, Y_test, verbose=0)
print ("score")
model.save('model_3app.h5') #saving the model
import pickle

with open('trainHistory3app', 'wb') as handle: # saving the history of the model
   pickle.dump(history3app.history, handle)
