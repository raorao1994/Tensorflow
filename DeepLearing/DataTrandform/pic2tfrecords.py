# -*- coding: utf-8 -*-
# 将图片转为TFRecord
import os.path
import matplotlib.image as mping
import tensorflow as tf
#import PIL import Image

SAVE_PATH='data/dataset.tfrecords'

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def resize(image_data,width,hight):
    return tf.image.resize_images(image_data,[width,hight],method=0)

def load_data(datafile,width,hight,save=False):
    train_list=open(datafile,'r')
    # 准备一个write用来写TFRecord文件
    writer=tf.python_io.TFRecordWriter(SAVE_PATH)

    with tf.Session() as sess:
        for line in train_list:
            # 获得图片的路径和类型
            tmp = line.strip().split(' ')
            img_path=tmp[0]
            label=int(tmp[1])

            # 读取图片
            image=tf.gfile.FastGFile(img_path,'r').read()
            # 解码图片（如果是png图片就是用decode_png）
            image=tf.image.decode_jpeg(image)
            # 转换数据类型
            # 因为为了将图片数据能够保存到 TFRecord 结构体中，所以需要将其图片矩阵转换成 string，所以为了在使用时能够转换回来，这里确定下数据格式为 tf.float32
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            # 既然都将图片保存成 TFRecord 了，那就先把图片转换成希望的大小吧
            image = resize(image, width, hight)
            # 执行 op: image
            image = sess.run(image)

            # 将其图片矩阵转换成 string
            image_raw = image.tostring()
            # 将数据整理成 TFRecord 需要的数据结构
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _byte_feature(image_raw),
                'label': _int64_feature(label),
            }))

            # 写 TFRecord
            writer.write(example.SerializeToString())

    # 关闭保存文件
    writer.close()

if __name__=="__main__":
    load_data('train_list.txt', 224, 224)
    print("运行完成！")