from keras_ocr import detection,tools
import numpy as np 
import matplotlib.pyplot as plt
import cv2

detector=detection.Detector(weights='C:/text-recognition/keras_ocr_vinAI.h5')

images= ['C:/text-recognition/test/im1167.jpg']

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

boxs=detector.detect(images)

print(boxs[0][1])

def drawBoxes(image, box, color=(255, 0, 0), thickness=5, boxes_format="boxes"):
    canvas = image.copy()
    cv2.polylines(
        img=canvas[0],
        pts=box[np.newaxis].astype("int32"),
        color=color,
        thickness=thickness,
        isClosed=True,
    )
    return canvas
drawn =tools.drawBoxes(
    image=images, boxes=boxs[0]
)

print(drawn[0])
cv2.imshow('windows', drawn[0])
cv2.waitKey(0)  
cv2.imwrite(f"C:/text-recognition/image_sample/6.jpg", drawn[0])