'''
转换图片大小
'''
import tensorflow as tf

def resize(image_data,width,hight):
    return tf.image.resize_images(image_data,[width,hight],method=0)
