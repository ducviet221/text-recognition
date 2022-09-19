from keras_ocr import datasets
import keras_ocr
from keras_ocr import datasets
import argparse
from keras_ocr.datasets import get_dataset
from keras_ocr.utils import preprocessing,training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--label', required=True, default='D:/text-recognition\dataset\labels", help='folder label')
    parser.add_argument('-i', '--image', required=True,default='D:/text-recognition\dataset\images,' help='folder image')
    parser.add_argument('-cp, --checkpoint', required=False, default='D:/text-recognition/detector_keras_vinAI', help='checkpoint path')
    parser.add_argument('-b, --batch_size', help='batch_size', default=1)
    parser.add_argument('-l, --log-file',default='C:/text-recognition/csv_VinAI.csv', help='CSVLogeer')
    parser.add_argument('-e, --epoch', help='epoch', default=1000, type = int)
    parser.add_argument('-p, --patience', help='patience', default=5, type =int)
    parser.add_argument('-r, --random_state', required=False, help='random_state', default=42, type = int)
    parser.add_argument('-t, --train_size', required=False, help='train_size', default=0.8, type = float)

    args = parser.parse_args()

    dataset = keras_ocr.datasets.dataset 
    train_img_gen, val_img_gen, val, train = preprocessing(dataset, args.random_state, args.label, args.image)
    # dataset=get_dataset(args.label,args.image,skip_illegible=False) 

    detector = keras_ocr.detection.Detector()

    training(args.batch_size, args.epoch,train_img_gen,val_img_gen, val, train, args.log_file, args.checkpoint, args.p)
if __name__ == '__main__':
    main()