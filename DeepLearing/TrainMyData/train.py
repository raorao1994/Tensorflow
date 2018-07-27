#http://www.cnblogs.com/wktwj/p/7227544.html
import tensorflow as tf;
import numpy as np;
import os;

class inputData():

    #1 读取图片文件
    def get_files(filename):
        class_train = []
        label_train = []
        for train_class in os.listdir(filename):
            for pic in os.listdir(filename+train_class):
                class_train.append(filename+train_class+'/'+pic)
                label_train.append(train_class)
        temp = np.array([class_train,label_train])
        temp = temp.transpose()
        #shuffle the samples
        np.random.shuffle(temp)
        #after transpose, images is in dimension 0 and label in dimension 1
        image_list = list(temp[:,0])
        label_list = list(temp[:,1])
        label_list = [int(i) for i in label_list]
        #print(label_list)
        return image_list,label_list

    #2 产生用于训练的批次
    def get_batches(image, label, resize_w, resize_h, batch_size, capacity):
        # convert the list of images and labels to tensor
        image = tf.cast(image, tf.string)
        label = tf.cast(label, tf.int64)
        queue = tf.train.slice_input_producer([image, label])
        label = queue[1]
        image_c = tf.read_file(queue[0])
        image = tf.image.decode_jpeg(image_c, channels=3)
        # resize
        image = tf.image.resize_image_with_crop_or_pad(image, resize_w, resize_h)
        # (x - mean) / adjusted_stddev
        image = tf.image.per_image_standardization(image)

        image_batch, label_batch = tf.train.batch([image, label],
                                                  batch_size=batch_size,
                                                  num_threads=64,
                                                  capacity=capacity)
        images_batch = tf.cast(image_batch, tf.float32)
        labels_batch = tf.reshape(label_batch, [batch_size])
        return images_batch, labels_batch

class model():
    #3 定义训练的模型
    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape,stddev = 0.01))
    # init weights
    weights = {
        "w1":init_weights([3,3,3,16]),
        "w2":init_weights([3,3,16,128]),
        "w3":init_weights([3,3,128,256]),
        "w4":init_weights([4096,4096]),
        "wo":init_weights([4096,2])
        }

    #init biases
    biases = {
        "b1":init_weights([16]),
        "b2":init_weights([128]),
        "b3":init_weights([256]),
        "b4":init_weights([4096]),
        "bo":init_weights([2])
        }

    #（2）定义不同层的操作
    def conv2d(x,w,b):
        x = tf.nn.conv2d(x,w,strides = [1,1,1,1],padding = "SAME")
        x = tf.nn.bias_add(x,b)
        return tf.nn.relu(x)
    def pooling(x):
        return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = "SAME")
    def norm(x,lsize = 4):
        return tf.nn.lrn(x,depth_radius = lsize,bias = 1,alpha = 0.001/9.0,beta = 0.75)

    #（3）定义训练模型
    def mmodel(images):
        l1 = conv2d(images,weights["w1"],biases["b1"])
        l2 = pooling(l1)
        l2 = norm(l2)
        l3 = conv2d(l2,weights["w2"],biases["b2"])
        l4 = pooling(l3)
        l4 = norm(l4)
        l5 = conv2d(l4,weights["w3"],biases["b3"])
        #same as the batch size
        l6 = pooling(l5)
        l6 = tf.reshape(l6,[-1,weights["w4"].get_shape().as_list()[0]])
        l7 = tf.nn.relu(tf.matmul(l6,weights["w4"])+biases["b4"])
        soft_max = tf.add(tf.matmul(l7,weights["wo"]),biases["bo"])
        return soft_max

    #（4）定义评估量
    def loss(logits,label_batches):
         cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=label_batches)
         cost = tf.reduce_mean(cross_entropy)
         return cost
    def get_accuracy(logits,labels):
        acc = tf.nn.in_top_k(logits,labels,1)
        acc = tf.cast(acc,tf.float32)
        acc = tf.reduce_mean(acc)
        return acc
    def training(loss,lr):
        train_op = tf.train.RMSPropOptimizer(lr,0.9).minimize(loss)
        return train_op

#4 训练
def run_training():
    data_dir = 'C:/Users/wk/Desktop/bky/dataSet/'
    image, label = inputData.get_files(data_dir)
    image_batches, label_batches = inputData.get_batches(image, label, 32, 32, 16, 20)
    p = model.mmodel(image_batches)
    cost = model.loss(p, label_batches)
    train_op = model.training(cost, 0.001)
    acc = model.get_accuracy(p, label_batches)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(1000):
            print(step)
            if coord.should_stop():
                break
            _, train_acc, train_loss = sess.run([train_op, acc, cost])
            print("loss:{} accuracy:{}".format(train_loss, train_acc))
    except tf.errors.OutOfRangeError:
        print("Done!!!")
    finally:
        coord.request_stop()
    coord.join(threads)
    sess.close()