from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
import time
import tensorflow.examples.tutorials.mnist.input_data as input_data
input_node=784
classes=10
layer1=100
layer2=50
batch_size=100

# import sys
# sys.path.append('..')
# from utils.layers import hidden_layer, DNN

tf.set_random_seed(2017)

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

train_set = mnist.train
test_set = mnist.test

# 定义模型
input_ph = tf.placeholder(shape=(None, 784), dtype=tf.float32)
label_ph = tf.placeholder(shape=(None, 10), dtype=tf.int64)

# dnn = DNN(input_ph, [200], weights_collection='params', biases_collection='params')
with tf.variable_scope('layer'):
    w1=tf.Variable(tf.random_normal([input_node, layer1], stddev=1, seed=1), name='w1')
    b1 = tf.Variable(tf.random_normal([layer1], stddev=1, seed=1), name='b1')
with tf.variable_scope('layer2'):
    w2 = tf.Variable(tf.random_normal([layer1, layer2], stddev=1, seed=1), name='w2')
    b2 = tf.Variable(tf.random_normal([layer2], stddev=1, seed=1), name='b2')
with tf.variable_scope('layer2'):
    w3 = tf.Variable(tf.random_normal([layer2, 10], stddev=1, seed=1), name='w3')
    b3 = tf.Variable(tf.random_normal([10], stddev=1, seed=1), name='b3')
with tf.variable_scope('two_network'):
    s1 = tf.nn.relu((tf.matmul(input_ph, w1)+b1))
    s2=tf.nn.relu((tf.matmul(s1, w2)+b2))
    dnn=tf.matmul(s2, w3)+ b3

# params = tf.get_collection('params',w1)
# params = tf.get_collection('params',w2)
# params = tf.get_collection('params',w3)
# params = tf.get_collection('params',b1)
# params = tf.get_collection('params',b2)
# params = tf.get_collection('params',b3)




loss = tf.losses.softmax_cross_entropy(logits=dnn, onehot_labels=label_ph)

acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(dnn, axis=-1), tf.argmax(label_ph, axis=-1)), dtype=tf.float32))


train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(loss)


with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    train_losses1 = []

    epoch = 0
    samples_passed = 0
    epoch_done = False
    step = 0

    _start = time.time()
    while (epoch < 5):
        if samples_passed + batch_size >= mnist.train.num_examples:
            this_batch = mnist.train.num_examples - samples_passed
            samples_passed = 0
            epoch += 1
            epoch_done = True
        else:
            samples_passed += batch_size
            this_batch = batch_size

        # 获取 batch_size个训练样本
        images, labels = train_set.next_batch(this_batch)
        if epoch_done:
            # 计算所有训练样本的损失值
            train_loss = []
            for _ in range(train_set.num_examples // 100):
                image, label = train_set.next_batch(100)
                loss_train = sess.run(loss, feed_dict={input_ph: image, label_ph: label})
                train_loss.append(loss_train)

            print('Epoch {} Train loss: {:.6f}'.format(epoch, np.array(train_loss).mean()))
            epoch_done = False

        # 每30步记录一次训练误差
        if step % 30 == 0:
            loss_train = sess.run(loss, feed_dict={input_ph: images, label_ph: labels})
            train_losses1.append(loss_train)

        sess.run(train_op, feed_dict={input_ph: images, label_ph: labels})
        step += 1

    _end = time.time()
    print('Train Done! Cost Time: {:.2f}s'.format(_end - _start))