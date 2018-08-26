'''
#This portion was used in kaggle kernel
import os
from os import listdir, makedirs
from os.path import join, exists, expanduser
from tqdm import tqdm

cache_dir = expanduser(join('~', '.keras'))
if not exists(cache_dir):
    makedirs(cache_dir)
models_dir = join(cache_dir, 'models')
if not exists(models_dir):
    makedirs(models_dir)

os.system('cp ../input/keras-pretrained-models/*notop* ~/.keras/models/')
os.system('cp ../input/keras-pretrained-models/imagenet_class_index.json ~/.keras/models/')
os.system('cp ../input/keras-pretrained-models/resnet50* ~/.keras/models/')
'''

import keras
from keras.applications import xception
from keras.optimizers import Adam
from keras.callbacks import *
from keras.models import Model, Sequential
from keras.layers import *
from keras import regularizers
from train_generator import TrainDataGenerator
from test_generator import TestDataGenerator
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from keras.preprocessing import image

TRAIN_PATH = 'train/'
TEST_PATH = 'test/'

base_model = xception.Xception(weights = 'imagenet', include_top = False, input_shape = (280, 280, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = Flatten()(x)

x = Dense(1024,  kernel_regularizer = regularizers.l2(0.05))(x)
x = Activation('relu')(x)
x = Dropout(0.28)(x)

x = Dense(1024, kernel_regularizer = regularizers.l2(0.05))(x)
x = Activation('relu')(x)
x = Dropout(0.28)(x)

predictions = Dense(120, activation = 'softmax', kernel_regularizer = regularizers.l2(0.05))(x)

model = Model(inputs = base_model.input, outputs = predictions)

model.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr=0.00001), metrics = ['accuracy'])

data = pd.read_csv('labels.csv')
data = data.values
classes = np.load('classes.npy')
mapping = {}
for i in range(classes.shape[0]):
    mapping[classes[i]] = i

labels = {}
for i in range(data.shape[0]):
    labels[data[i][0]] = mapping[data[i][1]]

ids = data[:,0]
ids = np.random.permutation(ids)
train_ids = ids[0:9500]
val_ids = ids[9500:]
num_val = len(val_ids)
num_train = len(train_ids)

training_generator = TrainDataGenerator(train_ids, labels, path = TRAIN_PATH, batch_size = 32, dim = (280, 280), n_channels = 3, n_classes = 120)
val_generator = TrainDataGenerator(val_ids, labels, path = TRAIN_PATH, batch_size = 32, dim = (280, 280), n_channels = 3, n_classes = 120)

model_checkpoint = ModelCheckpoint(filepath = 'weights.hdf5', verbose = 1, save_best_only = True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=2, min_lr=0.00002, verbose=1)
history = model.fit_generator(training_generator, epochs = 10, verbose = 2, validation_data = val_generator, callbacks = [reduce_lr])

#summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#Test the model

test_ids = np.load('test_ids.npy')
#Due to batch-size of 32, some ids will be left out
left_out_ids = test_ids[10336:]

test_set = TestDataGenerator(test_ids, path = TEST_PATH, batch_size = 32, dim = (280, 280), n_channels = 3)

X = np.empty((21, 280, 280, 3))
for i, id in enumerate(left_out_ids):
    img = image.load_img(TEST_PATH + id + '.jpg', target_size=(280, 280))
    X[i] = xception.preprocess_input(image.img_to_array(img))

left_out_set = X
predictions = model.predict_generator(test_set)
left_out_predictions = model.predict(left_out_set)
predictions = np.vstack((predictions, left_out_predictions))

print(predictions.shape)

#Prepare the submission file
data = {}
data['id'] = test_ids
for i in classes:
	data[i] = predictions[:, i]

df = pd.DataFrame(data)
df.to_csv('submission.csv')
