import  tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


INPUT_NODE = 784
OUTPUT_NODE = 10

LAYER1_NODE =500
BATCH_SIZE = 100

LEANING_RATE_BASE = 0.8
LEANING_RATE_DECAY = 0.99
REGULARZATION_RATE = 0.0001
TRAINING_STEPS =30000
MOVING_AVERAGE_DECAY = 0.99

# def inferense(input_tensor, avg_class, weight1, biases1, weight2, biases2):
#     if avg_class == None:
#         layer1 = tf.nn.relu(tf.matmul(input_tensor, weight1) + biases1)
#         return tf.matmul(layer1, weight2) + biases2
#
#     else:
#         layer1 = tf.nn.relu(
#             tf.matmul(input_tensor, avg_class.average(weight1)) + avg_class.average(biases1)
#         )
#         return tf.matmul(layer1, avg_class.average(weight2)) + avg_class.average(biases2)

def inference(input_tensor=None, reuse=False, avg_class=None):
    with tf.variable_scope('layer1', reuse=reuse):
        weights = tf.get_variable('weights', [INPUT_NODE, LAYER1_NODE],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable('weights', [LAYER1_NODE, OUTPUT_NODE],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2

def train(mnist):

    x = tf.placeholder(tf.float32, [None,INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')


    y = inference(x)
    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    average_y = inference(
        x, reuse=True, avg_class=variable_averages)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels= tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regularizer = tf.contrib.layers.l2_regularizer(REGULARZATION_RATE)
    with tf.variable_scope('layer1',reuse=True):
        w1 = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE])
    with tf.variable_scope('layer2',reuse=True):
        w2 = tf.get_variable("weights", [LAYER1_NODE, OUTPUT_NODE])
    regularization = regularizer(w1) + regularizer(w2)
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
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict= validate_feed)
                # test_acc = sess.run(accuracy,feed_dict=test_feed)
                print("After %d training step(s), validation accuracy "
                      "using avarage model is %g" % (i, validate_acc))
                # print(test_acc)

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average "
              "model is %g" % (TRAINING_STEPS, test_acc))

def main(argv=None):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()


