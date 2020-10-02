import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time
from detector import get_face

def getAllPath(dirpath, *suffix):
    PathArray = []
    for r, ds, fs in os.walk(dirpath):
        for fn in fs:
            if os.path.splitext(fn)[1] in suffix:
                fname = os.path.join(r, fn)
                PathArray.append(fname)
    return PathArray

if __name__ == "__main__":
    path = './lx'
    result = './result/' + path[1:]
    if not os.path.exists(result):
        os.makedirs(result)
    detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)
    ImagePaths=getAllPath(path, '.jpg','.JPG','png','PNG')
    for imagePath in ImagePaths:
        get_face(detector,imagePath)