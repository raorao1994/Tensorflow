import os,path;
import glob;
import random;
import numpy as np;
import  tensorflow as tf;
from tensorflow.python.platform import gfile;

#读取图片文件
#传入文件夹路径
def get_files(pathname):
    class_train=[];
    lable_train=[];
    #获取目录下的所有分类子目录
    for train_class_path in os.listdir(pathname):
        for pic in os.listdir(pathname+'/'+train_class_path):
            class_train.append(pic);
            lable_train.append(train_class_path);
    #创建数组对象
    temp=np.array([class_train,lable_train]);
    # 数组转置
    temp=temp.transpose();
    #数组打乱训练数据
    np.random.shuffle(temp);
    #转置后，图像在0维，标签在1维。
    #temp[:,0]表示获取第一维
    img_list=list(temp[:,0]);
    lable_list=list(temp[:,1]);
    #lable_list=[int(i) for i in lable_list];
    return img_list,lable_list;

if __name__ == '__main__':
    img_list,lable_list=get_files("E:/Github/projectdata/data/train_images");
    print(img_list);
    print(lable_list);
    print("完成");

