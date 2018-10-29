import cv2
import numpy as np
import os,sys;

#原图路径，mask图路径
srcImgPath="D:\\Tensorflow\\SegnetTrainData\\test\\1.png";
maskImgPath="D:\\Tensorflow\\SegnetTrainData\\predict\\1.png";

n_label=5
color_list={
    0:[159,255,84],
    1:[255,238,0],
    2:[34,180,238],
    3:[255,191,0],
    4:[38,71,139],
    5:[0,105,255],
    6:[35,0,255],
    7:[168,0,255],
    8:[255,0,255],
    9:[255,0,163],
    10:[129,170,117],
    11:[147,144,95]
};
#定义常量
VEGETATION=1
ROAD =4
BUILDING =2
WATER =3

def draw_lables():
    srcImg = cv2.imread(srcImgPath)
    srcHeight = srcImg.shape[0]  # 将tuple中的元素取出，赋值给height，width，channels
    srcWidth = srcImg.shape[1]
    maskImg = cv2.imread(srcImgPath,0)

    print("宽高:"+str(srcHeight)+":"+str(srcWidth))
    #遍历图像
    for i in range(srcHeight):
        if i%500==0:
            print("高:" + str(i))
        bigDada=0
        for j in range(srcWidth):
            p_src=srcImg[i][j]
            p_mask=maskImg[i][j]
            if p_mask>11:
                bigDada=bigDada+1
                p_mask=0

            color=color_list[p_mask]
            p_src[0] = color[0]
            p_src[1] = color[1]
            p_src[2] = color[2]


    #保存图像
    print("bigDada:"+str(bigDada))
    cv2.imwrite("result.png",srcImg)

if __name__=="__main__":
    print("运行程序")
    draw_lables();
    print("结束运行")