from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Tensorflow 已经把 mnist 数据集集成在 examples 里面了
# 在这里 import 数据输入的部分
import tensorflow.examples.tutorials.mnist.input_data as input_data

tf.set_random_seed(2017)
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_set = mnist.train
test_set = mnist.test


# 重置计算图
tf.reset_default_graph()

# 重新定义占位符
input_ph = tf.placeholder(shape=(None, 784), dtype=tf.float32)
label_ph = tf.placeholder(shape=(None, 10), dtype=tf.int64)

# 构造权重, 用`truncated_normal`初始化
def weight_variable(shape):
    init = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(init)

# 构造偏置, 用`0.1`初始化
def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


# 构造添加`variable`的`summary`的函数
def variable_summaries(var):
    with tf.name_scope('summaries'):
        # 计算平均值
        mean = tf.reduce_mean(var)
        # 将平均值添加到`summary`中, 这是一个数值, 所以我们用`tf.summary.scalar`
        tf.summary.scalar('mean', mean)

        # 计算标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 将标准差添加到`summary`中
        tf.summary.scalar('stddev', stddev)

        # 添加最大值,最小值`summary`
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))

        # 添加这个变量分布情况的`summary`, 我们希望观察它的分布, 所以用`tf.summary.histogram`
        tf.summary.histogram('histogram', var)


# 构造一个隐藏层
def hidden_layer(x, output_dim, scope='hidden_layer', act=tf.nn.relu, reuse=None):
    # 获取输入的`depth`
    input_dim = x.get_shape().as_list()[-1]

    with tf.name_scope(scope):
        with tf.name_scope('weight'):
            # 构造`weight`
            weight = weight_variable([input_dim, output_dim])
            # 添加`weight`的`summary`
            variable_summaries(weight)

        with tf.name_scope('bias'):
            # 构造`bias`
            bias = bias_variable([output_dim])
            # 添加`bias`的`summary`
            variable_summaries(bias)

        with tf.name_scope('linear'):
            # 计算`xw+b`
            preact = tf.matmul(x, weight) + bias
            # 添加激活层之前输出的分布情况到`summary`
            tf.summary.histogram('pre_activation', preact)

        # 经过激活层`act`
        output = act(preact)
        # 添加激活后输出的分布情况到`summary`
        tf.summary.histogram('output', output)
        return output
# 构造深度神经网络
def DNN(x, output_depths, scope='DNN_with_sums', reuse=None):
    with tf.name_scope(scope):
        net = x
        for i, output_depth in enumerate(output_depths):
            net = hidden_layer(net, output_depth, scope='hidden%d' % (i + 1), reuse=reuse)
        # 最后有一个分类层
        net = hidden_layer(net, 10, scope='classification', act=tf.identity, reuse=reuse)
        return net

dnn_with_sums = DNN(input_ph, [400, 200, 100])

# 重新定义`loss`, `acc`, `train_op`
with tf.name_scope('cross_entropy'):
    loss = tf.losses.softmax_cross_entropy(logits=dnn_with_sums, onehot_labels=label_ph)
    tf.summary.scalar('cross_entropy', loss)

with tf.name_scope('accuracy'):
    acc = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(dnn_with_sums, axis=-1), tf.argmax(label_ph, axis=-1)), dtype=tf.float32))
    tf.summary.scalar('accuracy', acc)

with tf.name_scope('train'):
    lr = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss)

merged = tf.summary.merge_all()
sess = tf.InteractiveSession()

train_writer = tf.summary.FileWriter('test_summary/train', sess.graph)
test_writer = tf.summary.FileWriter('test_summary/test', sess.graph)

batch_size = 64

sess.run(tf.global_variables_initializer())

for e in range(20000):
    images, labels = train_set.next_batch(batch_size)
    sess.run(train_op, feed_dict={input_ph: images, label_ph: labels})
    if e % 1000 == 999:
        test_imgs, test_labels = test_set.next_batch(batch_size)
        # 获取`train`数据的`summaries`以及`loss`, `acc`信息
        sum_train, loss_train, acc_train = sess.run([merged, loss, acc], feed_dict={input_ph: images, label_ph: labels})
        # 将`train`的`summaries`写入到`train_writer`中
        train_writer.add_summary(sum_train, e)
        # 获取`test`数据的`summaries`以及`loss`, `acc`信息
        sum_test, loss_test, acc_test = sess.run([merged, loss, acc], feed_dict={input_ph: test_imgs, label_ph: test_labels})
        # 将`test`的`summaries`写入到`test_writer`中
        test_writer.add_summary(sum_test, e)
        print('STEP {}: train_loss: {:.6f} train_acc: {:.6f} test_loss: {:.6f} test_acc: {:.6f}'.format(e + 1, loss_train, acc_train, loss_test, acc_test))

# 关闭读写器
train_writer.close()
test_writer.close()

print('Train Done!')
print('-'*30)

# 计算所有训练样本的损失值以及正确率
train_loss = []
train_acc = []
for _ in range(train_set.num_examples // 100):
    image, label = train_set.next_batch(100)
    loss_train, acc_train = sess.run([loss, acc], feed_dict={input_ph: image, label_ph: label})
    train_loss.append(loss_train)
    train_acc.append(acc_train)

print('Train loss: {:.6f}'.format(np.array(train_loss).mean()))
print('Train accuracy: {:.6f}'.format(np.array(train_acc).mean()))

# 计算所有测试样本的损失值以及正确率
test_loss = []
test_acc = []
for _ in range(test_set.num_examples // 100):
    image, label = test_set.next_batch(100)
    loss_test, acc_test = sess.run([loss, acc], feed_dict={input_ph: image, label_ph: label})
    test_loss.append(loss_test)
    test_acc.append(acc_test)

print('Test loss: {:.6f}'.format(np.array(test_loss).mean()))
print('Test accuracy: {:.6f}'.format(np.array(test_acc).mean()))