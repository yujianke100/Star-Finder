# coding: utf-8
import mxnet as mx
from detector.mtcnn_detector import MtcnnDetector
import cv2
import os
import time

def get_face(detector, imageName, imgPath, savePath):
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    img = cv2.imread(imgPath)
    unknowNum = 0
    # run detector
    results = detector.detect_face(img)
    if results is not None:
        total_boxes = results[0]
        points = results[1]
        # extract aligned face chips
        chips = detector.extract_image_chips(img, points, 112, 0.37)
        for i, chip in enumerate(chips):
            print("found the face!")
            # cv2.imshow('chip_'+str(i), chip)
            if(imageName == 'unknowFace'):
                cv2.imwrite(savePath+imageName+str(unknowNum)+'.jpg', chip)
                unknowNum += 1
            else:
                cv2.imwrite(savePath+imageName+'.jpg', chip)
        draw = img.copy()
        for b in total_boxes:
            cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))
        for p in points:
            for i in range(5):
                cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)
        return True
    else:
        return False