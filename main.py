import os
import cv2
import time
import shutil
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean 
from insightface.embedder import InsightfaceEmbedder
import mxnet as mx
from detector.mtcnn_detector import MtcnnDetector
from detector.detector import get_face

Persons = []

def getAllPath(dirpath, *suffix):
    PathArray = []
    for r, ds, fs in os.walk(dirpath):
        for fn in fs:
            if os.path.splitext(fn)[1] in suffix:
                fname = os.path.join(r, fn)
                PathArray.append(fname)
    return PathArray

def deleteFiles(path):
    rmList = os.listdir(path)
    for p in rmList:
        os.remove(path + p)
        print("delete:" + path + p)

def init():
    global embedder
    global Persons
    global ImagePaths
    global detectorModel
    originalPath = './images'
    finalPath = './knownFaces'
    detectorModel = MtcnnDetector(model_folder='./detector/model', ctx=mx.cpu(0), num_worker = 4 , accurate_landmark = False)
    ImagePaths=getAllPath(originalPath, '.jpg','.JPG','png','PNG')
    for imagePath in ImagePaths:
        imageName = imagePath[len(originalPath):-4]
        get_face(detectorModel,imageName, imagePath, finalPath)
    model_path = "models/model-y1-test2/model"
    embedder = InsightfaceEmbedder(model_path=model_path, epoch_num='0000', image_size=(112, 112))
    ImagePaths=getAllPath(finalPath, '.jpg','.JPG','png','PNG')
    Persons = []
    names = []
    for imagePath in ImagePaths:
        names.append(imagePath[len(finalPath)+1:-4])
        person = {'name':names[-1],'emb':embedder.embed_image(cv2.imread(imagePath))}
        Persons.append(person)
    print('\n\ninit compelete, and now I knowed:\n')
    with open(finalPath + "/list","w") as file:
        for name in names:
            print(name)
            file.write(name + '\n')
    with open("./lock","w") as file:
            file.write('0')

def faceCompare(cmpPath):
    global embedder
    global Persons
    global detectorModel
    deleteFiles('./result/')
    fid = open("./result/list",'w')
    fid.close()
    deleteFiles('./unknowFace/')
    get_face_result = get_face(detectorModel,'unknowFace', cmpPath, './unknowFace/')
    deleteFiles('./input/')
    if(not get_face_result):
        print("don't have face!")
    unknowImages = getAllPath('./unknowFace', '.jpg','.JPG','.png','.PNG', '.jfif')
    if not os.path.exists('./result'):
        os.makedirs('./result')
    with open("./result/list","w") as file:
        unknowSum = 0
        for unknowImage in unknowImages:
            cmpEmb = embedder.embed_image(cv2.imread(unknowImage))
            mixOne = {'name':'', 'cmp':2}
            for know in Persons:
                if(cmpEmb is None):
                    result = 999
                else:
                    result = euclidean(know['emb'], cmpEmb)
                if(result < mixOne['cmp']):
                    mixOne['cmp'] = result
                    mixOne['name'] = know['name']

            if(mixOne['cmp'] < 1):
                print('this is ' + mixOne['name'] + '! And the similarity degree is:' + str(mixOne['cmp']))
                try:
                    shutil.copy(unknowImage, './result/' + mixOne['name'] + '.jpg')
                    file.write(mixOne['name'] + '\n')
                except IOError as e:
                    print("Unable to copy file. %s" % e)

            else:
                print("I don't know who is it! And the similarity degree is:"+ str(mixOne['cmp']))
                try:
                    shutil.copy(unknowImage, './result/unknowFace' + str(unknowSum) + '.jpg')
                    file.write('unknowFace' + str(unknowSum) + '\n')
                    unknowSum += 1
                except IOError as e:
                    print("Unable to copy file. %s" % e)




def get_sign():
    while(True):    
        time.sleep(1.5)
        ImagePaths=getAllPath('./input', '.jpg','.JPG','.png','.PNG', '.jfif')
        if(ImagePaths):
            startTime = time.time();
            with open("./lock","w") as file:
                file.write('1')
            shutil.copy(ImagePaths[0], './default.jpg')
            return ImagePaths[0], startTime


if __name__ == "__main__":
    init()
    print('find_face is running!')
    while(True):
        cmpPath, startTime = get_sign()
        print(cmpPath)
        if(cmpPath == './input/exit.jpg'):
            deleteFiles('./input/')
            break
        faceCompare(cmpPath)
        endTime = time.time()
        with open("./time","w") as file:
            file.write(str(endTime - startTime))
        with open("./lock","w") as file:
            file.write('0')
        # time.sleep(0.5)
        with open("./result/list","a") as file:
            file.write('finished')
    print("Bye!")
