import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

mnist=input_data.read_data_sets("data",one_hot=True)

pic_size=28
classes=10
lr=0.01
channel=1

con1deep=32
con1size=5

con2deep=64
con2size=5
batch=100
fc=512

def inference():
    x=tf.placeholder(tf.float32,[None,pic_size,pic_size,channel],name='x')
    y=tf.placeholder(tf.float32,[None,classes],name='y')




    with tf.variable_scope('conv1'):
        con1_w=tf.get_variable('weight',[con1size,con1size,channel,con1deep],initializer=tf.truncated_normal_initializer(stddev=0.1))
        con1_b=tf.variable_scope('bias',[con1deep],initializer=tf.constant_initilizer(0.0))
        con1=tf.nn.conv2d(x,con1_w,strides=[1,1,1,1],padding='SAME')
        relu1=tf.nn.relu(tf.nn.bias_add(con1,con1_b))

    with tf.variable_scope('pooling1'):
        pool1=tf.nn.max_pool(relu1,ksize=[1,2,2,1],stride=[1,2,2,1],padding='SAME')

    with tf.variable_scope('conv2'):
        con2_w=tf.get_variable('weight',[con2size,con2size,con1deep,con2deep],initializer=tf.truncated_normal_initializer(stddev=0.1))
        con2_b=tf.variable_scope('bias',[con2deep],initializer=tf.constant_initilizer(0.0))
        con2=tf.nn.conv2d(pool1,con2_w,strides=[1,1,1,1],padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(con2,con2_b))

    with tf.variable_scope('pooling2'):
        pool2=tf.nn.max_pool(relu2,ksize=[1,2,2,1],stride=[1,2,2,1],padding='SAME')

    pool_shape=pool2.get_shape().as_list()
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    reshaped=tf.reshape(pool2,[pool_shape[0],nodes])


    with tf.variable_scope('fc1'):
        w1=tf.get_variable('weight',[nodes,fc],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b1=tf.get_variable('bias',[fc],initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc1=tf.nn.relu(tf.matmul(reshaped,w1)+b1)

    with tf.variable_scope('fc2'):
        w2=tf.get_variable('weight',[fc,classes],initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2=tf.get_variable('bias',[classes],initializer=tf.truncated_normal_initializer(stddev=0.1))
        y_pre=tf.matmul(fc1,w2)+b2
    with tf.variable_scope('soft_cross'):
        loss= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pre))
    with tf.variable_scope('acc'):
        acc=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pre,1), tf.argmax(y,1)),'float'))
    with tf.variable_scope('optimizer'):
        train_op = tf.train.RMSPropOptimizer(lr, 0.9).minimize(loss)
    loss_arr=[]
    acc_arr=[]


    init = tf.initialize_all_variables()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for e in range(15000):
            xs, ys = mnist.train.next_batch(batch)
            xs=np.reshape(xs,(batch,pic_size,pic_size,channel))
            _,loss_eval,acc_eval=sess.run([train_op,loss,acc],feed_dict={x:xs,y:ys})
            loss_arr.append(loss_eval)
            acc_arr.append(acc_eval)
            if (e+1)%1000==0:
                print('训练集第{}次训练，loss为{}，准确率为：{}'.format(e+1,loss_eval,acc_eval))
                xs_test, ys_test = mnist.test.next_batch(batch)
                xs_test = np.reshape(xs_test, (batch, pic_size, pic_size, channel))
                loss_eval, acc_eval=sess.run([loss,acc],feed_dict={x:xs_test,y:ys_test})
                print('测试集第{}次训练，loss为{}，准确率为：{}'.format(e + 1, loss_eval, acc_eval))

            if (e+1)%5000==0:
                saver.save(sess=sess,save_path='modelsave/model.ckpt',global_step=e+1)
    sess.close()
    plt.figure(1)
    plt.plot(loss_arr, label='loss')
    plt.plot(acc_arr, label='acc')
    plt.legend()
    plt.show()



