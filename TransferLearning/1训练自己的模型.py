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
    #获取图片文件数据
    img_list=list(temp[:,0]);
    #获取图片lable
    lable_list=list(temp[:,1]);
    #lable_list=[int(i) for i in lable_list];
    return img_list,lable_list;

#获取批次数据
#resize_w,resize_h为输出图像大小
#batch_size，capacity为批次大小和容量
def get_batches(image,lable,resize_w,resize_h,batch_size,capacity):
    #数据格式转换，转换成特定类型
    image=tf.cast(image,tf.string);
    lable=tf.case(lable,tf.int64);
    queue=tf.train.slice_input_producer([image,lable]);
if __name__ == '__main__':
    img_list,lable_list=get_files("E:/Github/projectdata/data/train_images");
    print(img_list);
    print(lable_list);
    print("完成");

