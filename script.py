import os
import math
import imgaug
#A library for image augmentation in machine learning experiments, 
#particularly convolutional neural networks. Supports the augmentation of images
#, keypoints/landmarks, bounding boxes, heatmaps and 
#segmentation maps in a variety of different ways.
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import tensorflow as tf
import typing
import kerasocr

data_dir = '.'

def get_dataset(training_gt_dir,training_images_dir,skip_illegible=False):
  dataset=[]
  image_files_name = os.listdir(training_images_dir)
  label_files_name = os.listdir(training_gt_dir)
  for image_file, label_file in zip(image_files_name,label_files_name):
        image_path = os.path.join(training_images_dir, image_file )
        lines = []
        with open(os.path.join(training_gt_dir, label_file), "r", encoding="utf8") as f:
            current_line: typing.List[typing.Tuple[np.ndarray, str]] = []
            for raw_row in f.read().split("\n"):
                if raw_row == "":
                    lines.append(current_line)
                    current_line = []
                else:
                    row = raw_row.split(" ")[4:]
                    #take from the 4th element onwards
                    character = row[-1][1:-1]
                    #the last element is the letter
                    if character == "" and skip_illegible:
                        continue
                    x1, y1, x2, y2 = map(float, row[:4])
                    #get 4 final coordinates of file label execpt last letter
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    #cast type to int
                    current_line.append(
                        (np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]), character)
                    )
        lines = [line for line in lines if line]
        dataset.append((image_path, lines, 1))
  return dataset

training_gt_dir="D:/text-recognition\dataset\labels"
training_images_dir="D:/text-recognition\dataset\images"

dataset=get_dataset(training_gt_dir,training_images_dir,skip_illegible=False)

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
training_image_generator = kerasocr.datasets.get_detector_image_generator(
    labels=train,
    augmenter=augmenter,
    **generator_kwargs
)
validation_image_generator = kerasocr.datasets.get_detector_image_generator(
    labels=validation,
    **generator_kwargs
)

detector = kerasocr.detection.Detector()

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
        tf.keras.callbacks.CSVLogger(os.path.join(data_dir, 'detector_icdar2013.csv')),
        #Callback that streams epoch results to a CSV file.
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(data_dir, 'D:/text-recognition/detector_icdar2013.h5'))
        #Callback to save the Keras model or model weights at some frequency.
    ],
    validation_data=validation_generator,
    validation_steps=math.ceil(len(validation) / batch_size)
)#val_loss: quantity to be monitored.