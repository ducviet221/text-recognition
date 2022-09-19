from keras_ocr import detection,tools
import numpy as np 

import cv2
import glob
import warnings
import pyclipper


detector=detection.Detector()
def drawBoxes(image, boxes, color=(255, 0, 0), thickness=5):
    # if len(boxes) == 0:
    #     return image
    canvas = image.copy()
    cv2.polylines(
        img=canvas[0],
        pts=boxes[np.newaxis].astype("int32"),
        color=color,
        thickness=thickness,
        isClosed=True,
    )

    return canvas

# images= ['D:/Python/OCR/Naver/vietnamese/vintext/train_images/im0029.jpg']
path = glob.glob('D:/Python/OCR/Naver/vietnamese/vintext/train_images/*')
# print(path)
path_image = 'D:/Python/OCR/Naver/vietnamese/vintext/train_images/im0002.jpg'
# index = 0

# for i in path:
images = [path_image]
images = [tools.read(image) for image in images]
images = [tools.resize_image(image,max_scale=2,max_size=2048) for image in images]
max_height, max_width = np.array(
            [image.shape[:2] for image, scale in images]
        ).max(axis=0)
scales = [scale for _, scale in images]
images = np.array(
        [
            tools.pad(image, width=max_width, height=max_height)
            for image, _ in images
        ]
    )
boxes=detector.detect(images)

# drawn =tools.drawBoxes(
#     image=images, boxes=boxes[0]
# )

# cv2.imshow('windows', drawn[0])
# cv2.waitKey(0)
# cv2.imwrite(f"D:/Python/OCR/text-recognition/result/test_{index}.jpg", drawn[0])
# index += 1
# for index in range(len(boxes)):
#     print(f"box {index}:",boxes[0][index]/2)
#     drawn_one = drawBoxes(images, boxes[0][index], color=(255, 0, 0), thickness=5)
#     cv2.imwrite(f"D:/Python/OCR/text-recognition/result/test_{index}.jpg", drawn_one[0])
#     index += 1

f = open("D:/Python/OCR/Naver/vietnamese_original/vietnamese/Images/labels/gt_2.txt", 'r', encoding="utf8")
lines = f.readlines()
for line in lines:
    line = line.split(',')
    
    for i in range(len(line)):
        if 'CỔNG' in line[i]:

            arr = [[int(line[0]), int(line[1])], [int(line[2]), int(line[3])],[int(line[4]), int(line[5])], [int(line[6]), int(line[7])]]
def iou_score(box1, box2):
    """Returns the Intersection-over-Union score, defined as the area of
    the intersection divided by the intersection over the union of
    the two bounding boxes. This measure is symmetric.

    Args:
        box1: The coordinates for box 1 as a list of (x, y) coordinates
        box2: The coordinates for box 2 in same format as box1.
    """
    if len(box1) == 2:
        x1, y1 = box1[0]
        x2, y2 = box1[1]
        box1 = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    if len(box2) == 2:
        x1, y1 = box2[0]
        x2, y2 = box2[1]
        box2 = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    if any(
        cv2.contourArea(np.array(box, dtype="int32")[:, np.newaxis, :]) == 0
        for box in [box1, box2]
    ):
        warnings.warn("A box with zero area was detected.")
        return 0
    pc = pyclipper.Pyclipper()
    pc.AddPath(np.array(box1, dtype="int32"), pyclipper.PT_SUBJECT, closed=True)
    pc.AddPath(np.array(box2, dtype="int32"), pyclipper.PT_CLIP, closed=True)
    intersection_solutions = pc.Execute(
        pyclipper.CT_INTERSECTION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD
    )
    union_solutions = pc.Execute(
        pyclipper.CT_UNION, pyclipper.PFT_EVENODD, pyclipper.PFT_EVENODD
    )
    union = sum(
        cv2.contourArea(np.array(points, dtype="int32")[:, np.newaxis, :])
        for points in union_solutions
    )
    intersection = sum(
        cv2.contourArea(np.array(points, dtype="int32")[:, np.newaxis, :])
        for points in intersection_solutions
    )
    return intersection / union   

if __name__ == "__main__":
    iou_list = []
    for i in range(len(boxes[0])):

        iou = iou_score(arr, boxes[0][i]/2)
        iou_list.append(iou)
    print('iou average: ', sum(iou_list)/len(iou_list))