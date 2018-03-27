import tensorflow as tf;
import numpy as np;
import os.path;
from tensorflow.contrib.tensorboard.plugins import projector;

#分类数
class_num=21;
#运行次数
max_steps=1001;
#图片数量
batch_size=50;
#文件路径
image_path="F:/下载资源/地类数据/UCMerced_LandUse/Images/";
#训练率
lr=0.001;
#模型保存路径
save_path="F:/下载资源/地类数据/UCMerced_LandUse/model/";
#tensorboard保存路径
board_path="F:/下载资源/地类数据/UCMerced_LandUse/log/";

#获取训练文件下的所有数据
#返回图片路径列表和标签列表
def get_AllImg(path):
    img_list=[];
    lable_list=[];
    for sub_dir in os.listdir(path):
        for img_name in os.listdir(path+sub_dir):
            lable_list.append(sub_dir);
            img_list.append(path+sub_dir+"/"+img_name);
    #创建临时数据数组
    temp=np.array([img_list,lable_list]);
    #数组转置
    temp=temp.transpose();
    #随机搅乱数据
    np.random.shuffle(temp);
    #重新获取数据
    img_list1=list(temp[:,0]);
    lable_list1=list(temp[:,1]);
    return img_list1,lable_list1;

#产生训练批次数据
def get_batchs(imgs,labs,resize_w,resize_h,batch_size,capacity):
    #转换tensorflow数据类型
    img=tf.cast(imgs,tf.string);
    lab=tf.cast(labs,dtype=tf.int64);
    #实现一个输入的队列。
    queue=tf.train.slice_input_producer([img,lab]);
    lable=queue[1];
    image=tf.read_file(queue[0]);
    image_c=tf.image.decode_jpeg(image,channels=3);
    #图片尺寸转换
    image_c=tf.image.resize_image_with_crop_or_pad(image_c,resize_w,resize_h);
    # (x - mean) / adjusted_stddev  调整后的stddev
    image_c = tf.image.per_image_standardization(image_c);
    #生成批次对象
    image_batch,lable_batch=tf.train.batch([image_c,lable],batch_size=batch_size,capacity=capacity);
    # 转化图片
    image_batch=tf.cast(image_batch,tf.float32);
    #重新定义下 label_batch 的形状
    lable_batch=tf.reshape(lable_batch,[batch_size]);
    return image_batch,lable_batch;

#初始化权重
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev = 0.01))
#init weights权重
weights = {
    #卷积盒的为 3*3 的卷积盒，图片厚度是3，输出是16个featuremap
    "w1":init_weights([3,3,3,16]),
    "w2":init_weights([3,3,16,128]),
    "w3":init_weights([3,3,128,256]),
    "w4":init_weights([4096,4096]),
    "wo":init_weights([4096,class_num])
    }

#init biases偏移量
biases = {
    "b1":init_weights([16]),
    "b2":init_weights([128]),
    "b3":init_weights([256]),
    "b4":init_weights([4096]),
    "bo":init_weights([class_num])
    }

#定义卷积池化操作
#采用relu激活函数
def conv2d(x,w,b):
    x=tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME");
    x=tf.nn.bias_add(x,b);
    return tf.nn.relu(x);

def pooling(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME");

#不知道这个是什么操作
#归一化操作？
def norm(x,lsize=4):
    return tf.nn.lrn(x,depth_radius=lsize,bias=1,alpha=0.001/9.0,beta=0.75);

#定义训练模型
def model(images):
    l1 = conv2d(images,weights["w1"],biases["b1"])
    l2 = pooling(l1)
    l2 = norm(l2)
    l3 = conv2d(l2,weights["w2"],biases["b2"])
    l4 = pooling(l3)
    l4 = norm(l4)
    l5 = conv2d(l4,weights["w3"],biases["b3"])
    #same as the batch size  与批次大小相同
    l6 = pooling(l5)
    l6 = tf.reshape(l6,[-1,weights["w4"].get_shape().as_list()[0]])
    l7 = tf.nn.relu(tf.matmul(l6,weights["w4"])+biases["b4"])
    soft_max = tf.add(tf.matmul(l7,weights["wo"]),biases["bo"])
    return soft_max

def model1(images, batch_size, n_classes):

    with tf.variable_scope('conv1') as scope:
     # 卷积盒的为 3*3 的卷积盒，图片厚度是3，输出是16个featuremap
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    with tf.variable_scope('pooling1_lrn') as scope:
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pooling1')
            norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    with tf.variable_scope('conv2') as scope:
                weights = tf.get_variable('weights',
                                          shape=[3, 3, 16, 16],
                                          dtype=tf.float32,
                                          initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
                biases = tf.get_variable('biases',
                                         shape=[16],
                                         dtype=tf.float32,
                                         initializer=tf.constant_initializer(0.1))
                conv = tf.nn.conv2d(norm1, weights, strides=[1, 1, 1, 1], padding='SAME')
                pre_activation = tf.nn.bias_add(conv, biases)
                conv2 = tf.nn.relu(pre_activation, name='conv2')
    # pool2 and norm2
    with tf.variable_scope('pooling2_lrn') as scope:
        norm2 = tf.nn.lrn(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME', name='pooling2')
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool2, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
    local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
    # local4
    with tf.variable_scope('local4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name='local4')

    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[128, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[n_classes],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name='softmax_linear')
    return softmax_linear

#定义评估
def loss(logits,lable_bathes):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits \
            (logits=logits, labels=lable_bathes, name='xentropy_per_example');
        loss = tf.reduce_mean(cross_entropy, name='loss')

        tf.summary.scalar(scope.name + '/loss', loss)

    return loss

#定义损失函数
def accuracy(logits,lable):
    acc=tf.nn.in_top_k(logits,lable,1);
    acc=tf.cast(acc,tf.float32);
    accuracy = tf.reduce_mean(acc)
    return accuracy;

#定义训练方式
def train(loss,lr):
    train_op=tf.train.RMSPropOptimizer(lr,0.9).minimize(loss);
    return train_op;

#训练模型
def run_training():
    #1、获取所有数据
    image,lable=get_AllImg(image_path);
    print("数据获取完成");
    #2、获取批次数据
    image_batches, label_batches =get_batchs(image,lable,40,40,batch_size,20);
    #3、cnn模型计算
    train_logits =model1(image_batches,batch_size,class_num);
    #4、模型评估
    cost = loss(train_logits, label_batches);
    #5、模型训练
    train_op=train(cost,lr);
    #6、验证
    acc=accuracy(train_logits,label_batches);

    # 初始化变量
    init = tf.global_variables_initializer();
    #创建sess
    sess = tf.Session();
    sess.run(init);
    for step in np.arange(max_steps):
        print("训练");
        _, train_acc, train_loss = sess.run([train_op, acc,cost ]);
        print("训练完成");
        #每100步打印一次
        if step % 100 == 0:
            print("loss:{} accuracy:{}".format(train_loss, train_acc))
    sess.close()
    print("结束");

if __name__ == '__main__':
    run_training();