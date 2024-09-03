###############################################
## CNN - Petliak                             ##
## RMB                                       ##
## April-May 2024                            ##
###############################################

################################################
# Package imports
################################################
import tensorflow as tf
print(tf.__version__)
import keras
from tensorflow.keras import datasets, layers, models
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import subprocess
import cv2

print(tf.config.list_physical_devices('GPU'))
print(tf.reduce_sum(tf.random.normal([1000, 1000])))

from sklearn.model_selection import train_test_split
import rasterio
import rasterio.features
import rasterio.warp
from rasterio.plot import show
from PIL import Image, ImageOps
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical # used to be: from keras.utils import np_utils; to develop one hot encoding
from tqdm import tqdm
from alive_progress import alive_bar
import time
import pandas as pd
from osgeo import gdal

###############################################
# Testing/visualizing training data
###############################################
fp = r'C:/Users/rmbcu/Documents/PetliakData/TrainingData/ALP.tif'
img = rasterio.open(fp)
img = img.read()

img = np.reshape(img, [3090, 2483])
img_plt = plt.imshow(img, cmap = 'terrain')

img1 = rasterio.open(r'C:/Users/rmbcu/Documents/PetliakData/TrainingData/CAL.tif')
show(img1, with_bounds = False, contour = False, title = 'CAL', adjust = None, cmap = 'gray')

img2 = rasterio.open(r'C:/Users/rmbcu/Documents/PetliakData/TrainingData/ELD.tif')
show(img2, with_bounds = False, contour = False, title = 'ELD', adjust = None, cmap = 'gray')

img3 = rasterio.open(r'C:/Users/rmbcu/Documents/PetliakData/TrainingData/FRS.tif')
show(img3, with_bounds = False, contour = False, title = 'FRS', adjust = None, cmap = 'gray')

img4 = rasterio.open(r'C:/Users/rmbcu/Documents/PetliakData/TrainingData/MAR.tif')
show(img4, with_bounds = False, contour = False, title = 'MAR', adjust = None, cmap = 'gray')

img5 = rasterio.open(r'C:/Users/rmbcu/Documents/PetliakData/TrainingData/PLA.tif')
show(img5, with_bounds = False, contour = False, title = 'PLA', adjust = None, cmap = 'gray')

img6 = rasterio.open(r'C:/Users/rmbcu/Documents/PetliakData/TrainingData/TUL.tif')
show(img6, with_bounds = False, contour = False, title = 'TUL', adjust = None, cmap = 'gray')

img7 = rasterio.open(r'C:/Users/rmbcu/Documents/PetliakData/TrainingData/TUO.tif')
show(img7, with_bounds = False, contour = False, title = 'TUO', adjust = None, cmap = 'gray')

img8 = rasterio.open(r'C:/Users/rmbcu/Documents/PetliakData/TrainingData/TUL2.tif')
show(img8, with_bounds = False, contour = False, title = 'TUL2', adjust = None, cmap = 'gray')

fig, axes = plt.subplots(ncols=4, nrows=2, figsize=(10, 4))
show(img1, cmap='binary', title = 'CAL', ax = axes[0][0])
show(img2, cmap='terrain', title = 'ELD', ax = axes[0][1])
show(img3, cmap='binary', title = 'FRS', ax = axes[0][2])
show(img4, cmap='binary', title = 'MAR', ax = axes[0][3])
show(img5, cmap='binary', title = 'PLA', ax = axes[1][0])
show(img6, cmap='binary', title = 'TUL', ax = axes[1][1])
show(img7, cmap='binary', title = 'TUO', ax = axes[1][2])
show(img8, cmap='binary', title = 'TUL2', ax = axes[1][3])

###############################################
# Read metadata for TIF files
###############################################

# TUL labels for training data
tul_map = gdal.Open('C:/Users/rmbcu/Documents/PetliakData/TrainingData/TUL.tif')
metadata_tul = gdal.Info(tul_map)
for line in metadata_tul.split('\n'): print(line)
# TUL mask
# Upper Left  (  392396.400, 3977559.600) (118d11'34.51"W, 35d56'11.29"N)
# Lower Left  (  392396.400, 3974599.800) (118d11'33.07"W, 35d54'35.25"N)
# Upper Right (  395411.400, 3977559.600) (118d 9'34.20"W, 35d56'12.47"N)
# Lower Right (  395411.400, 3974599.800) (118d 9'32.80"W, 35d54'36.42"N)
# Center      (  393903.900, 3976079.700) (118d10'33.65"W, 35d55'23.86"N)

#fml.tif - has no data area
#idk77.tif - does not have no data area
tul_baseAll = gdal.Open('C:/Users/rmbcu/Documents/PetliakData/fml.tif')
metadata_tul_baseAll = gdal.Info(tul_baseAll)
for line in metadata_tul_baseAll.split('\n'): print(line)
#tul_baseAll - has no data area
# Upper Left  (  392394.402, 3977559.996) (118d11'34.59"W, 35d56'11.31"N)
# Lower Left  (  392394.402, 3974597.331) (118d11'33.15"W, 35d54'35.16"N)
# Upper Right (  395411.366, 3977559.996) (118d 9'34.20"W, 35d56'12.49"N)
# Lower Right (  395411.366, 3974597.331) (118d 9'32.80"W, 35d54'36.34"N)
# Center      (  393902.884, 3976078.664) (118d10'33.69"W, 35d55'23.83"N)

tul_baseTUL = gdal.Open('C:/Users/rmbcu/Documents/PetliakData/idk77.tif')
metadata_tul_baseTUL = gdal.Info(tul_baseTUL)
for line in metadata_tul_baseTUL.split('\n'): print(line)
#tul_baseTUL - does not have 'no data' area
# Upper Left  (  933784.920, 3987756.335) (118d11'33.10"W, 35d56'16.44"N)
# Lower Left  (  933784.920, 3984664.542) (118d11'39.16"W, 35d54'36.44"N)
# Upper Right (  936920.731, 3987756.335) (118d 9'28.39"W, 35d56'11.42"N)
# Lower Right (  936920.731, 3984664.542) (118d 9'34.50"W, 35d54'31.42"N)
# Center      (  935352.826, 3986210.439) (118d10'33.79"W, 35d55'23.94"N)


###############################################
# Make TIF files into JPEGs - commented out 
# saves to prevent accidental overwrites
###############################################

### PROCEED w NAIP data TIF that includes the 'no data area' in the TUL mask map = fml.tif


im = Image.open('C:/Users/rmbcu/Documents/ArcGIS/Projects/Petliak_May24/idk3_5.tif')
im.thumbnail(im.size)
im.save('C:/Users/rmbcu/Documents/PetliakData/pil_tul_out_X.jpg', "JPEG", quality = 100)

### Had issues with converting the original TUL.tif file to JPEG
import tifffile
test = tifffile.imread('C:/Users/rmbcu/Documents/PetliakData/TrainingData/TUL.tif')
#cv2.imwrite('C:/Users/rmbcu/Documents/PetliakData/TrainingData/TUL_test.jpg', test)
# test = np.int8(test)
# Image.fromarray(test, mode="1") #other modes = L, P
# Image.fromarray(idk, mode="RGB").convert(mode='1', colors=1)

im = Image.open('C:/Users/rmbcu/Documents/PetliakData/TrainingData/TUL_test.jpg')
imarray = np.array(im)
plt.imshow(imarray, interpolation = 'none')
#plt.imshow(imarray, interpolation = 'antialiased')
#im.save('C:/Users/rmbcu/Documents/PetliakData/TrainingData/TUL_test_q100.jpg', "JPEG", quality = 100)

img = (np.maximum(test, 0) / test.max()) * 255.0
print(img)
#img = 255 - img 
img = Image.fromarray(np.uint8(img))
plt.imshow(img, interpolation = 'none')
#img.save('C:/Users/rmbcu/Documents/PetliakData/TrainingData/TUL_scale2.jpg', "JPEG", quality = 100)

#TUL_scale2 has '#img = 255 - img' commented out
#TUL_scale1 has all, based on test tifffile import from TUL.TIF 


###############################################
# Preprocessing
###############################################

# 65x65 input patch size

#For each pixel p being classified
# output = conditional probability that pixel is ROCK
# if prob < .50 --> NOT ROCK

img_x1 = rasterio.open(r'C:/Users/rmbcu/Documents/PetliakData/TrainingData/TUL.tif')
show(img_x1, with_bounds = False, contour = False, title = 'TUL Mask - original', adjust = None) #TUL_scale2.jpg looks mos like this

img_x2 = rasterio.open('C:/Users/rmbcu/Documents/ArcGIS/Projects/Petliak_May24/TUL_test2.tif')
show(img_x2, with_bounds = False, contour = False, title = 'TUL Mask - Matches the map', adjust = None) #TUL_scale2.jpg looks mos like this

img_x3 = Image.open('C:/Users/rmbcu/Documents/PetliakData/TrainingData/TUL_scale2.jpg')
imarray = np.array(img_x3)
plt.imshow(imarray)

img_x = rasterio.open('C:/Users/rmbcu/Documents/ArcGIS/Projects/Petliak_May24/Map_15.tif')
show(img_x, with_bounds = False, contour = False, title = 'TUL NAIP Map', adjust = None) #TUL_scale2.jpg looks mos like this
################################################################
################################################################
################################################################
# Whole area:
# TRAINING LABELS = TUL_scale2.jpg (OR TUL_test2.tif?)
# TRAINING DATA = Map_15.tif
import tifffile
trainMap = tifffile.imread('C:/Users/rmbcu/Documents/ArcGIS/Projects/Petliak_May24/Map_15.tif')
# (4933, 5025, 4) <--- THIS IS WHAT YOU NEED, 4 BANDS AND THE RIGHT SIZE!!!!!!

trainLabel = Image.open('C:/Users/rmbcu/Documents/PetliakData/TrainingData/TUL_scale2.jpg')
#(5025, 4933) 
trainLabel = np.array(trainLabel)
plt.imshow(trainLabel)
# (4933, 5025), looks yellowy

### USE THISONE FOR NOW ### 
trainLabel = tifffile.imread('C:/Users/rmbcu/Documents/ArcGIS/Projects/Petliak_May24/TUL_test2.tif')
trainLabel = np.array(trainLabel)
plt.imshow(trainLabel)

################################################################
################################################################
###############################################################

#now get the other one 
maskImg = tifffile.imread('C:/Users/rmbcu/Documents/ArcGIS/Projects/Petliak_May24/TUL_test2.tif')
np.shape(maskImg)
#(4934, 5026)

maskImg2 = tifffile.imread('C:/Users/rmbcu/Documents/PetliakData/TrainingData/TUL.tif')
np.shape(maskImg2)
#(4933, 5025)

maskImg3 = Image.open('C:/Users/rmbcu/Documents/PetliakData/TrainingData/TUL_test.jpg')
#(5025, 4933) THIS ONE?
maskImg3 = np.array(maskImg3)
plt.imshow(maskImg3)
# (4933, 5025)


img_x1_test = (np.maximum(img_x1, 0) / img_x1.max()) * 255.0
print(img)

#cv2.imwrite('C:/Users/rmbcu/Documents/PetliakData/TrainingData/TUL_test.jpg', test)
img = (np.maximum(test, 0) / test.max()) * 255.0
print(img)
#img = 255 
img = Image.fromarray(np.uint8(img))
plt.imshow(img, interpolation = 'none')
imgC=img.convert(mode = 'RGB')
imgC.save('C:/Users/rmbcu/Documents/PetliakData/TUL_inputMap.jpg', "JPEG", quality = 100)


img_temp = Image.open('C:/Users/rmbcu/Documents/PetliakData/TUL_inputMap.jpg')
img_temp_arr = np.array(img_temp)
plt.imshow(img_temp_arr)

trainMap = tifffile.imread('C:/Users/rmbcu/Documents/ArcGIS/Projects/Petliak_May24/Map_15.tif')
labelPath = 'C:/Users/rmbcu/Documents/PetliakData/TrainingData/TUL_scale2.jpg'

#########################################################
# 300x300px blocks
#########################################################
#tulMap_Img = Image.open('C:/Users/rmbcu/Documents/PetliakData/pil_tul_out1.jpg') # (889, 873) 
tulMap = tifffile.imread('C:/Users/rmbcu/Documents/ArcGIS/Projects/Petliak_May24/Map_15.tif')
#tulMap = np.array(tulMap)
#np.shape(tulMap) # (873, 889, 3)

tulLabels_Img = Image.open('C:/Users/rmbcu/Documents/ArcGIS/Projects/Petliak_May24/TUL_test2.tif') # (5025, 4933)
tulLabels = np.array(tulLabels_Img)
np.shape(tulMap) # (4933, 5025, 4)
np.shape(tulLabels) # (4934, 5026)
np.unique(tulMap)
np.unique(tulLabels)
np.shape(np.argwhere(tulLabels == 0))

totPix = 5025*4933
np.shape(np.argwhere(tulLabels == 0))[0] / totPix   # 44.51%
np.shape(np.argwhere(tulLabels == 128))[0] / totPix # 8.35%
np.shape(np.argwhere(tulLabels == 255))[0] / totPix # 47.17%
#44.51+8.35+47.17

testTUL = np.array(tulLabels_Img)
plt.imshow(testTUL)
np.where(testTUL == 0, 77, 1)
plt.imshow(np.where(testTUL == 0, 0, 255))
# 0 = purple
# 255 = yellow
# 128 = rock
testTUL[1000,3000]
testTUL[0,0]
testTUL[2000, 2000:2500]

np.argwhere(tulLabels == 128)
test = np.delete(tulLabels, [4933], axis=0)
test = np.delete(test, [5025], axis = 1)
np.shape(test)
#4933, 5025
tulLabels = test.copy()
testTUL = test.copy()

from skimage.util.shape import view_as_blocks

plt.imshow(tulLabels[50:4800, 200:4900])

labelShrink = tulLabels[50:4850, 100:4900]
mapShrink = tulMap[50:4850, 100:4900]
plt.imshow(mapShrink)
plt.imshow(labelShrink)
np.shape(labelShrink)
np.shape(mapShrink)

labelBlocks = view_as_blocks(labelShrink, (300,300))
mapBlocks =  view_as_blocks(mapShrink, (300,300, 4))
mapBlocks = np.reshape(mapBlocks, (16,16,300,300,4))

np.shape(labelBlocks) #(16, 16, 300, 300)
np.shape(mapBlocks) #(16,16, 300,300,4)

plt.imshow(labelBlocks[5,8])
plt.imshow(mapBlocks[5,8])

#########################################################
# 65x65px patches
#########################################################
block1_label = labelBlocks[5,5]
block1_map = mapBlocks[5,5]
def split_into_patches_with_shift(block, patch_size=65, shift=1):
    patches = []
    block_height, block_width = block.shape[:2]
    for y in range(0, block_height - patch_size + 1, shift):
        for x in range(0, block_width - patch_size + 1, shift):
            patch = block[y:y+patch_size, x:x+patch_size]
            patches.append(patch)
    return patches
patchLabels = split_into_patches_with_shift(block1_label)
np.shape(patchLabels) #(55696, 65, 65)

patchMap = split_into_patches_with_shift(block1_map)
np.shape(patchMap) #(55696, 65, 65, 4))

###############################################
# Prep data for CNN
###############################################
# patchMap = 55696 list of 65x65x4 px patches (map)
# patchLabels = 55696 list of 65x65px patches (labels)
from sklearn.model_selection import train_test_split
dataIn = np.array(patchMap)
labelsIn = np.array(patchLabels).reshape((np.shape(patchLabels)[0], 65, 65, 1))
np.shape(dataIn.T)
dataIn = dataIn.T
np.shape(labelsIn.T)
labelsIn = labelsIn.T

# Normalize
#dataIn = dataIn.astype('uint8')/255.0)
#labelsIn = np_utils.to_categorical(labelsIn)

# Split data into training and testing sets
train_patches, test_patches, train_labels, test_labels = train_test_split(dataIn, labelsIn, test_size = 0.35)

# def process_data_for_tensorflow(train_patches, test_patches, train_labels, test_labels):
#     # Convert lists to TensorFlow tensors
#     train_patches_tensor = tf.convert_to_tensor(train_patches, dtype=tf.uint8)
#     test_patches_tensor = tf.convert_to_tensor(test_patches, dtype=tf.uint8)
#     train_labels_tensor = tf.convert_to_tensor(train_labels, dtype=tf.uint8)
#     test_labels_tensor = tf.convert_to_tensor(test_labels, dtype=tf.uint8)

#     # Expand dimensions of label tensors if needed
#     if len(train_labels_tensor.shape) == 3:
#         train_labels_tensor = tf.expand_dims(train_labels_tensor, axis=-1)
#     if len(test_labels_tensor.shape) == 3:
#         test_labels_tensor = tf.expand_dims(test_labels_tensor, axis=-1)

#     return train_patches_tensor, test_patches_tensor, train_labels_tensor, test_labels_tensor

# # Process data for TensorFlow
# train_patches_tensor, test_patches_tensor, train_labels_tensor, test_labels_tensor = process_data_for_tensorflow(train_patches, test_patches, train_labels, test_labels)

# Print shapes for verification
# print("Train Patches Tensor Shape:", train_patches_tensor.shape)
# print("Test Patches Tensor Shape:", test_patches_tensor.shape)
# print("Train Labels Tensor Shape:", train_labels_tensor.shape)
# print("Test Labels Tensor Shape:", test_labels_tensor.shape)

xTrain = train_patches
xTest = test_patches
yTrain = train_labels
yTest = test_labels
###############################################
# Model Architecture
###############################################
# Parameters used in (Petliak 2019):
# ARCHITECTURE:
    # 9-layer convolutional neural network
    # Input = 4-band 65x65 NAIP imagery patch
    # 1st conv layer = 8@63x63, 8 3x3 filters zero-padding
    # Max pool 8@31x31 window 2x2
    # 2nd conv layer = 8@29x29, 8 3x3 filters zero-padding
    # Max pool 8@14x14 window 2x2
    # 3rd conv layer = 16@12x12, 16 filters
    # Max pool 16@6x6, window 2x2
    # Dense layer 1x32
    # Output = 1x1
# HYPERPARAMETERS:
#   OPTIMIZER: Adam
#   LEARNING RATE = 0.001
#   LOSS = Binary Cross-entropy
#   EPOCHS = 10
#   Mini-batch
# Define hyperparameters here
cnn_opt = 'adam'
lr = 0.001
loss_type = 'binary_crossentropy'
# lossType = tf.keras.losses.BinaryCrossentropy()
#Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), groups=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs)
#data_format: string, either "channels_last" or "channels_first". The ordering of the dimensions in the inputs. "channels_last" corresponds to inputs with shape (batch_size, height, width, channels) while "channels_first" corresponds to inputs with shape (batch_size, channels, height, width).
model = Sequential()
# Input = 4-band 65x65 NAIP imagery patch
model.add(Conv2D(filters = 8, 
                 kernel_size = (3, 3), 
                 input_shape = (65, 65, 4),
                 data_format='channels_last',
                 padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
# 2nd conv layer = 8@29x29, 8 3x3 filters zero-padding
model.add(Conv2D(filters = 8, 
                 kernel_size = (3, 3),
                 padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))

# 3rd conv layer = 16@12x12, 16 filters
model.add(Conv2D(filters = 16, 
                 kernel_size = (3, 3),
                 padding = 'same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.summary()

# Compile model
model.compile(loss = loss_type, 
              optimizer = cnn_opt, 
              metrics = ['accuracy'])
###############################################
# Training
###############################################
# Parameters used in (Petliak 2019):
#   EPOCHS: 10
#       (CNNs trained 2hrs/epoch for 20 hours total-> 10 epochs?)
#   Mini-batch
#model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), batch_size=64)
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, shuffle=True)
history_list = []
for train_index, test_index in kf.split(xTrain):
    x1, x2 = train_index[0], train_index[-1]
    t1, t2 = test_index[0], test_index[-1]
    
    x_train_fold, x_val_fold = xTrain[x1:x2], xTrain[t1:t2]
    y_train_fold, y_val_fold = yTrain[x1:x2], yTrain[t1:t2]
    
    history = model.fit(x_train_fold, y_train_fold, 
                        epochs=10, 
                        validation_data=(x_val_fold, y_val_fold), 
                        batch_size=1,
                        verbose=1) 
    
    history_list.append(history.history)

history_df = pd.DataFrame()
for i, fold_history in enumerate(history_list):
    fold_history_df = pd.DataFrame(fold_history)
    fold_history_df['fold'] = i + 1
    history_df = history_df.append(fold_history_df, ignore_index=True)

# Plot training history
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
history_df[['loss', 'val_loss']].plot(title='Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(1, 2, 2)
history_df[['accuracy', 'val_accuracy']].plot(title='Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()


