from PIL import Image
import time
import argparse
from models.vietocr.tool.utils import vietocr
from scripts.crop import crop_image
from models.craft import file_utils
from models.craft.craft_utils import link_refiner, load_craft_net, str2bool, copyStateDict, test_net

def main():

    parser = argparse.ArgumentParser(description='CRAFT Text Detection')
    parser.add_argument('--trained_model', default='./models/weights/craft_mlt_25k.pth', type=str, help='pretrained model')
    parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
    parser.add_argument('--low_text', default=0.4, type=float, help='text low-bound score')
    parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
    parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda for inference')
    parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
    parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
    parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
    parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
    parser.add_argument('--test_folder', default='./models/craft/data', type=str, help='folder path to input images')
    parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
    parser.add_argument('--refiner_model', default='./models/weights/craft_refiner_CTW1500.pth', type=str, help='pretrained refiner model')

    args = parser.parse_args()
# load net
    net = load_craft_net(args.trained_model, args.cuda)
# LinkRefiner
    refine_net = link_refiner(args.refine, args.refiner_model, args.cuda, args.poly)
    t = time.time()
# load data
    """ For test images in a folder """
    file_utils.load_data(net, args.test_folder, args.canvas_size, args.mag_ratio, args.text_threshold, args.link_threshold, args.low_text, args.cuda, args.poly, args.show_time, refine_net)

    print("elapsed time : {}s".format(time.time() - t))

#Read image files
    img_files = file_utils.read_image_file()
    # print(img_files)
#Crop
    crop_image(img_files)
    
#models vietocr
    vietocr(img_files)


if __name__ == '__main__':
    main()
