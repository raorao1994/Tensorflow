#coding=utf-8

import cv2
import random
import os
import numpy as np
from tqdm import tqdm

img_w = 256  
img_h = 256  

basePath="D:\\Tensorflow\\SegnetTrainData\\base2\\";
savePath="D:\\Tensorflow\\SegnetTrainData\\train\\";

image_sets = []

#读取所有文件
def readImgNames():
    for root,dirs,files in os.walk(basePath+"src"):
        for file in files:
            image_sets.append(file)

def resizeImg():
    for file in image_sets:
        srcImg=cv2.imread(basePath+"src\\"+file)
        labelImg = cv2.imread(basePath + "label\\" + file)

        srcImg1=cv2.resize(srcImg,(256,256))
        labelImg1 = cv2.resize(labelImg, (256, 256))

        cv2.imwrite(savePath+"src\\"+file,srcImg1)
        cv2.imwrite(savePath + "label\\" + file, labelImg1)



if __name__=='__main__':
    print("开始");
    readImgNames()
    print(image_sets)
    resizeImg()
    print("完成")
