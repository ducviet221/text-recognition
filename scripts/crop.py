import numpy as np
import cv2
import os
from scripts.coors_get import len_coor_file, get_coors, coor_files
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
def crop_img(img_files, points):

    img = cv2.imread(img_files)
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    #method 1 smooth region
    cv2.drawContours(mask, [points], -1, (255, 255, 255), -1, cv2.LINE_AA)
    #method 2 not so smooth region
    # cv2.fillPoly(mask, points, (255))
    res = cv2.bitwise_and(img,img,mask = mask)
    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    return cropped

def crop_image(img_files):
    directory = './result/result_croped'
    len_coor = len_coor_file()
    # print(img_files)
    os.chdir(directory)
    for i in range(len(img_files)):
        array = []
        final_array = []
        # print(len_coor[i])
        for x in range(len_coor[i]):
            # print(len_coor)
            array.append(get_coors(coor_files[i],x).tolist())
        sort(array)
        final_array = np.array(array)
        for x in range(len_coor[i]):
            img = crop_img(img_files[i], final_array[x])
            file_name = str(i) + str(x) + '.jpg'
            cv2.imwrite(file_name, img)