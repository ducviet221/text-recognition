from PIL import Image
import os
import time
import argparse
import torch
import cv2 
import numpy as np
from crop import crop_img
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from coors_get import len_coor_file, get_coors, coor_files
from CRAFT.craft import CRAFT_func
from CRAFT import imgproc
from CRAFT import file_utils
from test_craft import str2bool, copyStateDict, test_net

def sort(arr):
    """_summary_

    Args:
        arr (_type_): _description_
    """
    n = len(arr)
	# Traverse through all array elements
    for i in range(n-1):
		# Last i elements are already in place
        for j in range(0, n-i-1):
			# traverse the array from 0 to n-i-1
			# Swap if the element found is greater
			# than the next element
            if arr[j][0] > arr[j + 1][0] :
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
def main():
#import CRAFT
    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default=r'D:\Python\OCR\text-recognition\model\craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default=r'D:\Python\OCR\text-recognition\CRAFT\data', type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='D:\Python\OCR\text-recognition\model\craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

    args = parser.parse_args()
# load net
    net = CRAFT_func()     # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    net.eval()
# LinkRefiner
    refine_net = None
    if args.refine:
        from CRAFT.refinenet import RefineNet
        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()
# load data
    """ For test images in a folder """
    image_list, _, _ = file_utils.get_files(args.test_folder)

    result_folder = './result/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k+1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        bboxes, polys, score_text = test_net(net, image, args.canvas_size, args.mag_ratio, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, args.show_time, refine_net)

        file_utils.saveResult(image_path, image[:,:,::-1], polys, dirname=result_folder)

    print("elapsed time : {}s".format(time.time() - t))

#Read image files
    img_files = []
    for (dirpath, dirnames, filenames) in os.walk(r'D:\Python\OCR\text-recognition\result'):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
#Crop
    directory = r'D:\Python\OCR\text-recognition\result_final'
    len_coor = len_coor_file()
    print(len_coor)
    os.chdir(directory)
    for i in range(len(img_files)):
        array = []
        final_array = []
        for x in range(len_coor[i]):
            array.append(get_coors(coor_files[i],x).tolist())
        sort(array)
        final_array = np.array(array)
        for x in range(len_coor[i]):
            img = crop_img(img_files[i], final_array[x])
            file_name = str(i) + str(x) + '.jpg'
            cv2.imwrite(file_name, img)
        
#VietOCR
    config = Cfg.load_config_from_name('vgg_transformer')

    config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA'
    config['cnn']['pretrained']=False
    config['device'] = 'cpu'
    config['predictor']['beamsearch']=False

    detector = Predictor(config)
    img_files.clear()
    for (dirpath, dirnames, filenames) in os.walk(r'D:\Python\OCR\text-recognition\result_final'):
        for file in filenames:
            filename, ext = os.path.splitext(file)
            ext = str.lower(ext)
            if ext == '.jpg' or ext == '.jpeg' or ext == '.gif' or ext == '.png' or ext == '.pgm':
                img_files.append(os.path.join(dirpath, file))
    for x in img_files:
        img = Image.open(x)

        result = detector.predict(img, return_prob=True)

        print(result)


if __name__ == '__main__':
    main()
