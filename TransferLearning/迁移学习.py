import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

'''ģ�ͼ�����·������'''

BOTTLENECK_TENSOR_SIZE = 2048                          # ƿ����ڵ����
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'           # ƿ���������������
JPEG_DATA_TENSOR_NAME  = 'DecodeJpeg/contents:0'       # �������������

MODEL_DIR  = './inception_dec_2015'                    # ģ�ʹ���ļ���
MODEL_FILE = 'tensorflow_inception_graph.pb'           # ģ����

CACHE_DIR  = './bottleneck'                            # ƿ�������ת�ļ���
INPUT_DATA = './flower_photos'                         # �����ļ���

VALIDATION_PERCENTAGE = 10                             # ��֤�����ݰٷֱ�
TEST_PERCENTAGE       = 10                             # ���������ݰٷֱ�

'''����������粿��������'''

LEARNING_RATE = 0.01
STEP          = 4000
BATCH         = 100

def creat_image_lists(validation_percentage,testing_percentage):
    '''
    ��ͼƬ(��·���ļ���)��Ϣ�������ֵ���
    :param validation_percentage: ��֤���ݰٷֱ� 
    :param testing_percentage:    �������ݰٷֱ�
    :return:                      �ֵ�{��ǩ:{�ļ���:str,ѵ��:[],��֤:[],����:[]},...}
    '''
    result = {}
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # ����os.walk()�б��һ����'./'�������ų�
    is_root_dir = True            #<-----
    # ��������label�ļ���
    for sub_dir in sub_dirs:
        if is_root_dir:           #<-----
            is_root_dir = False
            continue

        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list  = []
        dir_name   = os.path.basename(sub_dir)
        # �����������ܵ��ļ�β׺
        for extension in extensions:
            # file_glob = os.path.join(INPUT_DATA,dir_name,'*.'+extension)
            file_glob = os.path.join(sub_dir, '*.' + extension)
            file_list.extend(glob.glob(file_glob))      # ƥ�䲢�ռ�·��&�ļ���
            # print(file_glob,'\n',glob.glob(file_glob))
        if not file_list: continue

        label_name = dir_name.lower()                   # ����label��ʵ�ʾ���Сд�ļ�����

        # ��ʼ������·��&�ļ��ռ�list
        training_images   = []
        testing_images    = []
        validation_images = []

        # ȥ·����ֻ�����ļ���
        for file_name in file_list:
            base_name = os.path.basename(file_name)

            # ����������ݸ���֤�Ͳ���
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (validation_percentage + testing_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
        # ����ǩ�ֵ�������
        result[label_name] = {
            'dir'        : dir_name,
            'training'   : training_images,
            'testing'    : testing_images,
            'validation' : validation_images
        }
    return result

def get_random_cached_bottlenecks(sess,n_class,image_lists,batch,category,jpeg_data_tensor,bottleneck_tensor):
    '''
    ���������ȡһ��batch��ͼƬ��Ϊѵ������
    :param sess: 
    :param n_class: 
    :param image_lists: 
    :param how_many: 
    :param category:            training or validation
    :param jpeg_data_tensor: 
    :param bottleneck_tensor: 
    :return:                    ƿ��������� & label
    '''
    bottlenecks   = []
    ground_truths = []
    for i in range(batch):
        label_index = random.randrange(n_class)              # ��ǩ�����������
        label_name  = list(image_lists.keys())[label_index]  # ��ǩ����ȡ
        image_index = random.randrange(65536)                # ��ǩ��ͼƬ�����������
        # ƿ��������
        bottleneck = get_or_create_bottleneck(               # ��ȡ��Ӧ��ǩ���ͼƬƿ������
            sess,image_lists,label_name,image_index,category,
            jpeg_data_tensor,bottleneck_tensor)
        ground_truth = np.zeros(n_class,dtype=np.float32)
        ground_truth[label_index] = 1.0                      # ��׼���[0,0,1,0...]
        # �ռ�ƿ��������label
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks,ground_truths

def get_or_create_bottleneck(
        sess,image_lists,label_name,index,category,jpeg_data_tensor,bottleneck_tensor):
    '''
    Ѱ���Ѿ������ұ�����������������������Ҳ������ȼ����������������Ȼ�󱣴浽�ļ�
    :param sess: 
    :param image_lists:       ȫͼ���ֵ�
    :param label_name:        ��ǰ��ǩ
    :param index:             ͼƬ����
    :param category:          training or validation
    :param jpeg_data_tensor: 
    :param bottleneck_tensor: 
    :return: 
    '''
    label_lists  = image_lists[label_name]          # ����ǩ�ֵ��ȡ ��ǩ:{�ļ���:str,ѵ��:[],��֤:[],����:[]}
    sub_dir      = label_lists['dir']               # ��ȡ��ǩֵ
    sub_dir_path = os.path.join(CACHE_DIR,sub_dir)  # �����ļ�·��
    if not os.path.exists(sub_dir_path):os.mkdir(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists,label_name,index,category)
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        #image_data = gfile.FastGFile(image_path,'rb').read()
        image_data = open(image_path,'rb').read()
        # print(gfile.FastGFile(image_path,'rb').read()==open(image_path,'rb').read())
        # ������ǰ�������ƿ������
        bottleneck_values = run_bottleneck_on_images(sess,image_data,jpeg_data_tensor,bottleneck_tensor)
        # list2string�Ա���д���ļ�
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        # print(bottleneck_values)
        # print(bottleneck_string)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    # ���ص���listע��
    return bottleneck_values

def run_bottleneck_on_images(sess,image_data,jpeg_data_tensor,bottleneck_tensor):
    '''
    ʹ�ü��ص�ѵ���õ�Inception-v3ģ�ʹ���һ��ͼƬ���õ����ͼƬ������������
    :param sess:              �Ự���
    :param image_data:        ͼƬ�ļ����
    :param jpeg_data_tensor:  �����������
    :param bottleneck_tensor: ƿ���������
    :return:                  ƿ������ֵ
    '''
    # print('input:',len(image_data))
    bottleneck_values = sess.run(bottleneck_tensor,feed_dict={jpeg_data_tensor:image_data})
    bottleneck_values = np.squeeze(bottleneck_values)
    # print('bottle:',len(bottleneck_values))
    return bottleneck_values

def get_bottleneck_path(image_lists, label_name, index, category):
    '''
    ��ȡһ��ͼƬ����ת��featuremap����ַ(���txt)
    :param image_lists:   ȫͼƬ�ֵ�
    :param label_name:    ��ǩ��
    :param index:         ���������
    :param category:      training or validation
    :return:              ��ת��featuremap����ַ(���txt)
    '''
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'

def get_image_path(image_lists, image_dir, label_name, index, category):
    '''
    ͨ��������ơ��������ݼ���ͼƬ��Ż�ȡһ��ͼƬ����ת��featuremap����ַ(��txt)
    :param image_lists: ȫͼƬ�ֵ�
    :param image_dir:   ����ļ��У��ڲ��Ǳ�ǩ�ļ��У�
    :param label_name:  ��ǩ��
    :param index:       ���������
    :param category:    training or validation
    :return:            ͼƬ�м������ַ
    '''
    label_lists   = image_lists[label_name]
    category_list = label_lists[category]       # ��ȡĿ��categoryͼƬ�б�
    mod_index     = index % len(category_list)  # �����ȡһ��ͼƬ������
    base_name     = category_list[mod_index]    # ͨ��������ȡͼƬ��
    return os.path.join(image_dir,label_lists['dir'],base_name)

def get_test_bottlenecks(sess,image_lists,n_class,jpeg_data_tensor,bottleneck_tensor):
    '''
    ��ȡȫ���Ĳ�������,�������
    :param sess: 
    :param image_lists: 
    :param n_class: 
    :param jpeg_data_tensor: 
    :param bottleneck_tensor: 
    :return:                   ƿ����� & label
    '''
    bottlenecks  = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    for label_index,label_name in enumerate(image_lists[label_name_list]):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]): # ����, {�ļ���}
            bottleneck = get_or_create_bottleneck(
                sess, image_lists, label_name, index,
                category, jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_class, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths

def main():
    # �����ļ��ֵ�
    images_lists = creat_image_lists(VALIDATION_PERCENTAGE,TEST_PERCENTAGE)
    # ��¼label����(�ֵ�����)
    n_class = len(images_lists.keys())

    # ����ģ��
    # with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_FILE),'rb') as f:   # �Ķ���������
    with open(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:            # �Ķ���������
        graph_def = tf.GraphDef()                                         # ����ͼ
        graph_def.ParseFromString(f.read())                               # ͼ����ģ��
    # ����ͼ�Ͻڵ�����(���վ�����)
    bottleneck_tensor,jpeg_data_tensor = tf.import_graph_def(             # ��ͼ�϶�ȡ������ͬʱ����Ĭ��ͼ
        graph_def,
        return_elements=[BOTTLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME])

    '''�µ�������'''
    # �����,��ԭģ�������feed
    bottleneck_input   = tf.placeholder(tf.float32,[None,BOTTLENECK_TENSOR_SIZE],name='BottleneckInputPlaceholder')
    ground_truth_input = tf.placeholder(tf.float32,[None,n_class]               ,name='GroundTruthInput')
    # ȫ���Ӳ�
    with tf.name_scope('final_train_ops'):
        Weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE,n_class],stddev=0.001))
        biases  = tf.Variable(tf.zeros([n_class]))
        logits  = tf.matmul(bottleneck_input,Weights) + biases
        final_tensor = tf.nn.softmax(logits)
    # ��������ʧ����
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=ground_truth_input))
    # �Ż��㷨ѡ��
    train_step    = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)

    # ��ȷ��
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor,1),tf.argmax(ground_truth_input,1))
        evaluation_step    = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(STEP):
            # ���batch��ȡƿ����� & label
            train_bottlenecks,train_ground_truth = get_random_cached_bottlenecks(
                sess,n_class,images_lists,BATCH,'training',jpeg_data_tensor,bottleneck_tensor)
            sess.run(train_step,feed_dict={bottleneck_input:train_bottlenecks,ground_truth_input:train_ground_truth})

            # ÿ����100������һ����֤����
            if i % 100 == 0 or i + 1 == STEP:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_class, images_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    bottleneck_input: validation_bottlenecks, ground_truth_input: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' %
                      (i, BATCH, validation_accuracy * 100))

        test_bottlenecks,test_ground_truth = get_test_bottlenecks(
            sess,images_lists,n_class,jpeg_data_tensor,bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step,feed_dict={
            bottleneck_input:test_bottlenecks,ground_truth_input:test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))

if __name__ == '__main__':
    main()