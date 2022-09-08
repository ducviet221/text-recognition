import os
import math
import re
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
import argparse
from keras_ocr.datasets import get_dataset
from keras_ocr.image_preprocessing import preprocessing,training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--label', required=True, help='file label')
    parser.add_argument('-i', '--image', required=True, help='file image')
    parser.add_argument('--checkpoint', required=False, help='checkpoint', default='D:/text-recognition/detector_keras_vinAI')
    parser.add_argument('--batch_size', help='batch_size', default=1)
    parser.add_argument('--CSVLogger', help='CSVLogeer', default='C:/text-recognition/csv_VinAI.csv')
    parser.add_argument('--epoch', help='epoch', default=1000, type = int)
    parser.add_argument('--patience', help='patience', default=5, type =int)
    parser.add_argument('--random_state', required=False, help='random_state', default=42, type = int)
    parser.add_argument('--train_size', required=False, help='train_size', default=0.8, type = float)
    parser.add_argument('--p', help='patience', default=5, type =int)

    args = parser.parse_args()

    dataset = keras_ocr.datasets.dataset 
    train_img_gen, val_img_gen, val, train = preprocessing(dataset, args.random_state, args.label, args.image)
    # dataset=get_dataset(args.label,args.image,skip_illegible=False) 

    detector = keras_ocr.detection.Detector()

    training(args.batch_size, args.epoch,train_img_gen,val_img_gen, val, train, args.CSVLogger, args.checkpoint, args.p)
if __name__ == '__main__':
    main()