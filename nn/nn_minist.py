import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist=input_data.read_data_sets("../tmp/data",one_hot=True)

input_node=784
classes=10
layer1=100
layer2=50
batch=100
lr=0.02


##############data

def full_connected():
    with tf.variable_scope('data'):
        x = tf.placeholder(tf.float32, [None, 784])
        y = tf.placeholder(tf.float32, [None, 10])

    ###定义前向传播过程
    with tf.variable_scope('layer1'):
        w1 = tf.Variable(tf.random_normal([input_node, layer1], stddev=1, seed=1), name='w1')  # 数据可能一个一个输入的
        b1 = tf.Variable(tf.random_normal([layer1], stddev=1, seed=1), name='b1')

    with tf.variable_scope('layer2'):
        w2 = tf.Variable(tf.random_normal([layer1, 10], stddev=1, seed=1), name='w2')
        b2 = tf.Variable(tf.random_normal([10], stddev=1, seed=1), name='b2')

    with tf.variable_scope('two_network'):
        s1 = tf.sigmoid((tf.matmul(x, w1)+b1))
        y_pre=tf.matmul(s1, w2)+ b2

    with tf.variable_scope('soft_cross'):
        loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pre))
    with tf.variable_scope('optimizer'):
        train_op=tf.train.GradientDescentOptimizer(learning_rate=lr, name='optimizer').minimize(loss)
# loss=tf.losses.softmax_cross_entropy(logits=y_pre,onehot_labels=y)####
# loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_pre)))   #求交叉熵
# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_pre, 1))
# loss = tf.reduce_mean(cross_entropy)


    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(y,1))   #在测试阶段，测试准确度计算
    acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    loss_arr=[]
    acc_arr=[]
    ################train
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        for e in range(15000):
        # datafeed={x:mnist.validation.images,y:mnist.validation.labels}
            xs,ys=mnist.train.next_batch(batch)
            # loss_arr.append(loss_numpy)
            _,loss_val,accc=sess.run([train_op,loss,acc],feed_dict={x:xs,y:ys})
            loss_arr.append(loss_val)
            acc_arr.append(accc)
            if e%500==0:
                print(loss_val)

                # y_pre_numpy=y_pre.eval(session=sess)
                # print(y_pre_numpy)
                # y_true_label=y.eval(session=sess)
                # y_pre_labe=(y_pre_numpy)
                # acc=np.mean(y_pre_labe==y_true_label)
                print('step is ',(e,loss_val,accc))

    sess.close()
    plt.plot(loss_arr, label='loss')
    plt.plot(acc_arr, label='acc')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    full_connected()

