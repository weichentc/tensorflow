# _*_ coding: utf-8 _*_
import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32
CONV1_SIZE = 5
# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE =5
# 全连接层的节点个数
FC_SIZE = 512


BATCH_SIZE = 100

LEANING_RATE_BASE = 0.8
LEANING_RATE_DECAY = 0.99
REGULARZATION_RATE = 0.0001
TRAINING_STEPS =30000
MOVING_AVERAGE_DECAY = 0.99
VALIDATE_SIZE = 5000
TEST_SIZE = 10000
def inference(input_tensor, train=True, regularizer=None,reuse=tf.AUTO_REUSE, avg_class=None):
    with tf.variable_scope('layer1-conv1', reuse=tf.AUTO_REUSE ):
        conv1_weights = tf.get_variable(
            'weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable(
            'biases', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        # 第一层使用边长为5深度为32的过滤器，过滤器移动的步长为1，且使用全0填充。
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1,1,1,1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 第二层池化层，选用最大池化层，过滤器边长2，全0填充且移动步长为2。这一层输入是上一层输出，
    # 也就是28x28x32的矩阵，输出为14x14x32的矩阵。
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(
            relu1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    # 第三层为卷积层，输入为14x14x32的矩阵。输出为14x14x64的矩阵。
    with tf. variable_scope('layer3-conv2',reuse=tf.AUTO_REUSE):
        conv2_weights = tf.get_variable(
            'weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable(
            'biases', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))


        # 使用边长为5深度为64的过滤器，过滤器移动的步长为1，且使用全0填充。
        conv2 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1,1,1,1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 第四层为池化层，最大池化层，过滤器边长为2，全0填充且移动步长为2。输入为
    # 14x14x64，输出为7x7x64。
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(
            relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    pool_shape = pool2.get_shape().as_list()

    #计算将矩阵拉直成向量之后的长度。
    # pool_shape[0]为一个batch中数据的个数。
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    #通过tf.reshape函数将第四层的输出变成一个batch的向量。
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 第五层全连接层，输入为向量，长度3136，输出向量长度512。
    with tf.variable_scope('layer5-fc1',reuse=tf.AUTO_REUSE):
        fc1_weights = tf.get_variable(
            'weight', [nodes, FC_SIZE],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable(
            'bias', [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train: fc1 = tf.nn.dropout(fc1, 0.5)

    #第六层全连接层，输入512的向量，输出长度为10的向量。这一层输出通过Softmax之后得到最后的结果。
    with tf.variable_scope('layer6-fc2',reuse=tf.AUTO_REUSE):
        fc2_weights = tf.get_variable(
            'weight', [FC_SIZE, NUM_LABELS],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable(
            'bias', [NUM_LABELS],
            initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    # 返回第六层的输出
    return logit


def train(mnist):

    x = tf.placeholder(tf.float32, [BATCH_SIZE,
                                    IMAGE_SIZE,
                                    IMAGE_SIZE,
                                    NUM_CHANNELS], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')


    y = inference(x)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(
        x)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels= tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARZATION_RATE)
    with tf.variable_scope('layer1-conv1',reuse=tf.AUTO_REUSE):
        w1 = tf.get_variable("weights", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP])
    with tf.variable_scope('layer3-conv2',reuse=tf.AUTO_REUSE):
        w2 = tf.get_variable("weights", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP])
    with tf.variable_scope('layer5-fc1',reuse=tf.AUTO_REUSE):
        w3 = tf.get_variable("weight", [6272, FC_SIZE])
    with tf.variable_scope('layer6-fc2',reuse=tf.AUTO_REUSE):
        w4 = tf.get_variable("weight", [FC_SIZE, NUM_LABELS])
    regularization = regularizer(w1) + regularizer(w2) + regularizer(w3) + regularizer(w4)
    loss = cross_entropy_mean + regularization
    learnig_rate = tf.train.exponential_decay(LEANING_RATE_BASE, global_step,
                                              mnist.train.num_examples / BATCH_SIZE, LEANING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate=learnig_rate) \
        .minimize(loss, global_step=global_step)
    # with tf.control_dependencies([train_step, variables_averages_op]):
    #     train_op = tf.no_op(name='train')
    train_op = tf.group(train_step, variables_averages_op)
    correct_prediction = tf.equal(tf.argmax(average_y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_x,ys1 = mnist.validation.next_batch(BATCH_SIZE)
        test_x,ys2 = mnist.test.next_batch(BATCH_SIZE)
        reshaped_validate_x = np.reshape(validate_x,(BATCH_SIZE,
                                                     IMAGE_SIZE,
                                                     IMAGE_SIZE,
                                                     NUM_CHANNELS))
        # reshaped_test_x = np.reshape(test_x, (BATCH_SIZE,
        #                                       IMAGE_SIZE,
        #                                       IMAGE_SIZE,
        #                                       NUM_CHANNELS))
        validate_feed = {x: reshaped_validate_x, y_:ys1}
        # test_feed = {x: reshaped_test_x, y_: ys2}
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict= validate_feed)
                # test_acc = sess.run(accuracy,feed_dict=test_feed)
                print("After %d training step(s), validation accuracy "
                      "using avarage model is %g" % (i, validate_acc))
                # print(test_acc)

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            reshaped_xs = np.reshape(xs, (BATCH_SIZE,
                                      IMAGE_SIZE,
                                      IMAGE_SIZE,
                                      NUM_CHANNELS))
            sess.run(train_op, feed_dict={x:reshaped_xs, y_: ys})

        # test_acc = sess.run(accuracy, feed_dict=test_feed)
        # print("After %d training step(s), test accuracy using average "
        #       "model is %g" % (TRAINING_STEPS, test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()

