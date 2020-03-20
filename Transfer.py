# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 10:12:23 2020

@author: user
"""

from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.applications.inception_v3 import InceptionV3
import os
import matplotlib.pyplot as plt

path = os.getcwd()

print(os.listdir(path+"/training/crime/train"))

print(os.listdir(path+"/training/crime/test"))


img_width, img_height = 150, 150

train_data_dir = path+'/training/crime/train'

validation_data_dir = path+'/training/crime/test'

test_data_dir = path+'/training/crime/test'

nb_train_samples = 3889
nb_validation_samples = 127
epochs = 2
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


base_model=InceptionV3(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(120,activation='softmax')(x) #final layer with softmax activation
model=Model(inputs=base_model.input,outputs=preds)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255, shear_range=0.2,
    zoom_range=0.2, horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)


train_generator = train_datagen.flow_from_directory(
    train_data_dir, target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir, target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='binary')

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height), batch_size=batch_size,
    class_mode='binary')


history = model.fit_generator( train_generator,
    steps_per_epoch=nb_train_samples // batch_size, epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

#Save the model file
model_json = model.to_json()
with open("Model/model.json", "w") as json_file:
    json_file.write(model_json)

#Save the weight files
model.save_weights('Model/weight.h5')

scores = model.evaluate_generator(test_generator)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()