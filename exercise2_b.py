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
from keras import models
from sklearn.metrics import confusion_matrix
from pickle import load
import matplotlib.pyplot as plt
import pandas as pd
import random


def getIndexPositions(listOfElements, element):
#Returns the indexes of all occurrences of give element in the list- listOfElements 
    indexPosList = []
    indexPos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            indexPos = listOfElements.index(element, indexPos)
            # Add the index position in list
            indexPosList.append(indexPos)
            indexPos += 1
        except ValueError as e:
            break
    return indexPosList


def list_of_images():
    SVHN_directory = os.path.join(os.path.join(os.getcwd(), os.path.join("svhn", "test.mat")))
    # load .mat file
    data_raw = loadmat(SVHN_directory)
    data = np.array(data_raw['X'])

    # make correct shape
    data = np.moveaxis(data, -1, 0)
    data = data.astype('float32')
    data /= 255
    
    labels = data_raw['y']
    labels[labels == 10] = 0
    
    index=np.zeros(10)
    list_of_images=[]
    for i in range (0,10):
        index_positions=getIndexPositions(list(labels), i)
        """
        label=np.array([1,5,7,3,8,0,2,4,9,6])
        condition=label==i
        print(condition)
        index_positions=np.extract(condition, label)
        print(index_positions)
        """
        index[i]= random.choice(index_positions)
        print(labels[int(index[i])])
        list_of_images.append(data[int(index[i])])
    print(index)
    return list_of_images
          
    
def plot_outputs(model, no_of_layers, image , digit):

    layer_outputs = [layer.output for layer in model.layers[:no_of_layers]] # Extracts the outputs of the top 12 layers
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs) # Creates a model that will return these outputs, given the model input
    image_reshaped= image.reshape(1, 32, 32,3)

    activations = activation_model.predict(image_reshaped) # Returns a list of six Numpy arrays: one array per layer activation 
    
    layer_names = []
   

    plt.imshow(image)
    plt.show()


    for layer in model.layers[:no_of_layers]:
        layer_names.append(layer.name) # Names of the layers, so you can have them as part of your plot
      
    images_per_row = 9
     
    fig, axs = plt.subplots(no_of_layers,1,gridspec_kw={'hspace': 1.6})
    fig.suptitle('By layer output for each convolutional filter. Digit : ' + str(digit))

    grids=[] 
    
    for layer_name, layer_activation in zip(layer_names, activations): # Displays the feature maps
        
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                
                channel_image = layer_activation[0,
                                             :, :,
                                             col * images_per_row + row]
                channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
               
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, # Displays the grid
                             row * size : (row + 1) * size] = channel_image
        grids.append(display_grid)
        scale = 1. / size
    
    for ax in axs:
    
        index=list(axs).index(ax)
        ax.grid(False)
        ax.imshow(grids[index], aspect='auto')
        ax.xaxis.set_tick_params(labelsize=8)
        ax.set_title(layer_names[index], fontsize='small')
     
    plt.show()
        
# Recreate the exact same model, including its weights and the optimizer from exercise2_a
model = keras.models.load_model('model_2a.h5')

with open('trainHistory_2a', 'rb') as handle: # loading old history 
    oldhstry = load(handle) 
  
images=list_of_images()

for i in range(0,len(images)):
    
    plot_outputs(model, 6, images[i] , i)
   