'''
使用tensorflow实现VGG16网络结构
'''
import tensorflow as tf
import numpy as np
import TFRecord

#定义网络参数
learning_rate=0.001
display_step=5
epochs=10
keep_prob=0.5
class_num=2

#定义各种类型层

#定义卷积层
def conv_op(input_op,name,kh,kw,n_out,dh,dw):
    input_op=tf.convert_to_tensor(input_op)
    n_in=input_op.get_shape()[-1].value;
    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(scope+"w",
                               shape=[kh,kw,n_in,n_out],
                               dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initailizer_conv2d())
        conv=tf.nn.conv2d(input_op,kernel,(1,dh,dw,1),padding='SAME')
        bias_init_val=tf.constant(0.0,shape=[n_out],dtype=tf.float32)
        biases=tf.Variable(bias_init_val,trainable=True,name='b')
        z=tf.nn.bias_add(conv,bias_init_val)
        activation=tf.nn.relu(z,name=scope)
        return activation;

#定义全连接层
def fc_op(input_op,name,n_out):
    n_in=input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel=tf.get_variable(scope+'w',
                               shape=[n_in,n_out],
                               dtype=tf.float32,
                               initializer=tf.contrib.layers.xavier_initailizer())
        biase_init_val=tf.constant(0.1,shape=[n_out],dtype=tf.float32)
        biases=tf.Variable(biase_init_val,name='b')
        activation=tf.nn.relu(input_op,kernel,biases,name=scope)
        return activation;

#定义池化层
def maxpool_op(input_op,name,kh,kw,dh,dw):
    return tf.nn.max_pool(input_op,
                          ksize=[1,kh,kw,1],
                          strides=[1,dh,dw,1],
                          padding='SAME',
                          name=name)

#网络结构
def inference_op(input_op,keep_prob):
    # block 1 -- outputs 112x112x64
    conv1_1=conv_op(input_op,name='conv1_1',kh=3,kw=3,n_out=64,dh=1,dw=1)
    conv1_2=conv_op(conv1_1,name='conv1_2',kh=3,kw=3,n_out=64,dh=1,dw=1)
    pool1=maxpool_op(conv1_2,name='pool1',kh=2,kw=2,dw=2,dh=2)

    # block 2 -- outputs 56x56x128
    conv2_1 = conv_op(pool1, name="conv2_1", kh=3, kw=3, n_out=128, dh=1, dw=1)
    conv2_2 = conv_op(conv2_1, name="conv2_2", kh=3, kw=3, n_out=128, dh=1, dw=1)
    pool2 = maxpool_op(conv2_2, name="pool2", kh=2, kw=2, dh=2, dw=2)

    # # block 3 -- outputs 28x28x256
    conv3_1 = conv_op(pool2, name="conv3_1", kh=3, kw=3, n_out=256, dh=1, dw=1)
    conv3_2 = conv_op(conv3_1, name="conv3_2", kh=3, kw=3, n_out=256, dh=1, dw=1)
    conv3_3 = conv_op(conv3_2, name="conv3_3", kh=3, kw=3, n_out=256, dh=1, dw=1)
    pool3 = maxpool_op(conv3_3, name="pool3", kh=2, kw=2, dh=2, dw=2)

    # block 4 -- outputs 14x14x512
    conv4_1 = conv_op(pool3, name="conv4_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv4_2 = conv_op(conv4_1, name="conv4_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv4_3 = conv_op(conv4_2, name="conv4_3", kh=3, kw=3, n_out=512, dh=1, dw=1)
    pool4 = maxpool_op(conv4_3, name="pool4", kh=2, kw=2, dh=2, dw=2)

    # block 5 -- outputs 7x7x512
    conv5_1 = conv_op(pool4, name="conv5_1", kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv5_2 = conv_op(conv5_1, name="conv5_2", kh=3, kw=3, n_out=512, dh=1, dw=1)
    conv5_3 = conv_op(conv5_2, name="conv5_3", kh=3, kw=3, n_out=512, dh=1, dw=1)
    pool5 = maxpool_op(conv5_3, name="pool5", kh=2, kw=2, dw=2, dh=2)

    #flatten
    shp=pool5.get_shape()
    flattened_shape=shp[1].value*shp[2].value*shp[3].value;
    resh1=tf.reshape(pool5,[-1,flattened_shape],name='resh1')

    #fully connected
    fc6=fc_op(resh1,name='fc6',n_out=4096)
    fc6_drop=tf.nn.dropout(fc6,keep_prob,name='fc6_drop')

    fc7=fc_op(fc6_drop,name='fc7',n_out=4096)
    fc7_drop=tf.nn.dropout(fc7,keep_prob,name='fc7_drop')

    logits=fc_op(fc7_drop,name='fc8',n_out=class_num)
    return logits;

#训练
def train(logits,labels):
    cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels)
    cost=tf.reduce_mean(cross_entropy)

    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    correct_pred=tf.equal(tf.arg_max(logits,1),tf.arg_max(labels,1))

    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    return optimizer,cost,accuracy;

#读取TFrecord
def readBatch(filename,batchsize):
    image_batch=0
    label_batch=0
    return image_batch,label_batch;

#运行程序
if __name__=="__main__":
    train_filename="train.tfrecords"
    test_filename="test.tfrecords"
    image_batch,label=readBatch(filename=train_filename,batchsize=2)
    test_image,test_label=readBatch(filename=test_filename,batchsize=20)

    pred=inference_op(input_op=image_batch,keep_prob=keep_prob)
    test_pred=inference_op(input_op=test_image,keep_prob=keep_prob)
    optimizer,cost,accuracy=train(logits=pred,labels=label)
    test_optimizer, test_cost, test_accuracy = train(logits=test_pred, labels=test_label)
    initop=tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
    with tf.Session as sess:
        sess.run(initop)
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(sess=sess,coord=coord)
        step=0
        while step<epochs:
            step+=1
            print(step)

            _,loss,acc=sess.run([optimizer,cost,accuracy])
            if step%display_step ==0:
                print(loss,acc)
        print('train finish!')

        _, testLoss, testAcc = sess.run([test_optimizer, test_cost, test_accuracy])
        print("Test acc = " + str(testAcc))
        print("Test Finish!")

