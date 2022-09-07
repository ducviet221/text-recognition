import os
import math
import imgaug
#A library for image augmentation in machine learning experiments, 
#particularly convolutional neural networks. Supports the augmentation of images
#, keypoints/landmarks, bounding boxes, heatmaps and 
#segmentation maps in a variety of different ways.
import numpy as np
import matplotlib.pyplot as plt
from keras_ocr import datasets
import sklearn.model_selection
import tensorflow as tf
import typing
import keras_ocr
from keras_ocr import datasets

data_dir = '.'
dataset = keras_ocr.datasets.dataset 
train, validation = sklearn.model_selection.train_test_split(
    dataset, train_size=0.8, random_state=42
)
augmenter = imgaug.augmenters.Sequential([
    imgaug.augmenters.Affine(
      scale=(1.0, 1.2),
      rotate=(-5, 5)
      # Apply affine transformations to each image.
      # Scale/zoom them, translate/move them, rotate them and shear them.
    ),
    imgaug.augmenters.GaussianBlur(sigma=(0, 0.5)),
    #Strengthen or weaken the contrast in each image.
    imgaug.augmenters.Multiply((0.8, 1.2), per_channel=0.2)
    # Make some images brighter and some darker.
    # In 20% of all cases, we sample the multiplier once per channel,
    # which can end up changing the color of the images.
])
generator_kwargs = {'width': 640, 'height': 640}
training_image_generator = keras_ocr.datasets.get_detector_image_generator(
    labels=train,
    augmenter=augmenter,
    **generator_kwargs
)
validation_image_generator = keras_ocr.datasets.get_detector_image_generator(
    labels=validation,
    **generator_kwargs
)

detector = keras_ocr.detection.Detector()

batch_size = 1
training_generator, validation_generator = [
    detector.get_batch_generator(
        image_generator=image_generator, batch_size=batch_size
    ) for image_generator in
    [training_image_generator, validation_image_generator]
]
detector.model.fit_generator(
    generator=training_generator,
    steps_per_epoch=math.ceil(len(train) / batch_size),
    epochs=1000,
    workers=0,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5),
        #patince: number of epochs with no improvement after which training will be stopped
        #the loss for 5 consecutive epochs.
        tf.keras.callbacks.CSVLogger(os.path.join(data_dir, 'detector_keras_ocr.csv')),
        #Callback that streams epoch results to a CSV file.
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(data_dir, 'D:/text-recognition/detector_keras_vinAI'))
        #Callback to save the Keras model or model weights at some frequency.
    ],
    validation_data=validation_generator,
    validation_steps=math.ceil(len(validation) / batch_size)
)#val_loss: quantity to be monitored.