import tensorflow as tf;
import numpy as np;
import os;
import platform as plt;
import cv2;

def get_one_image(img_dir):
    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([32, 32])
    image_arr = np.array(image)
    return image_arr


def test(test_file):
    log_dir = 'C:/Users/wk/Desktop/bky/log/'
    image_arr = get_one_image(test_file)

    with tf.Graph().as_default():
        image = tf.cast(image_arr, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 32, 32, 3])
        print(image.shape)
        p = model.mmodel(image, 1)
        logits = tf.nn.softmax(p)
        x = tf.placeholder(tf.float32, shape=[32, 32, 3])
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success')
            else:
                print('No checkpoint')
                prediction = sess.run(logits, feed_dict={x: image_arr})
                max_index = np.argmax(prediction)
                print(max_index)