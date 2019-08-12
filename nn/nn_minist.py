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
def lookdata():
    train_set=mnist.train
    fig,axes=plt.subplots(ncols=6,nrows=2)
    plt.tight_layout()
    images,labels=train_set.next_batch(12,shuffle=False)
    for ind,(image,label) in enumerate(zip(images,labels)):
        image=image.reshape((28,28))
        row=ind//6
        col=ind%6
        axes[row][col].imshow(image,cmap="gray")
        axes[row][col].axis('off')
        [[a]]=np.argwhere(label==1)
        axes[row][col].set_title(a)

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
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for e in range(151):
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
            if e%5000==0:
                saver.save(sess=sess,save_path='modelsave/model.ckpt',global_step=e)

    sess.close()
    plt.figure(2)
    plt.plot(loss_arr, label='loss')
    plt.plot(acc_arr, label='acc')
    plt.legend()
    plt.show()

    sess=tf.Session()
    saver = tf.train.import_meta_graph('./modelsave/model.ckpt-15000.meta',clear_devices = True)
    saver.restore(sess, './modelsave/model.ckpt-15000')
    print(w1.eval(session=sess))
def tenbodsum():
    tf.reset_default_graph()
    input_ph=tf.placeholder(shape=(None,784),dtype=tf.float32)
    label_ph=tf.placeholder(shape=(None,10),dtype=tf.float64)
    def weight_variable(shape):
        init=tf.truncated_normal(shape=shape,stddev=0.1)
        return tf.Variable(init)
    def bias_variable(shape):
        init=tf.constant(0.1,shape=shape)
        return tf.Variable(init)
    def variable_summaries(var):
        with tf.name_scope('summaries'):
            mean=tf.reduce_mean(var)
            tf.summary.scalar('mean',mean)
            with tf.name_scope('stddev'):
                stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
            tf.summary.scalar('stddev',stddev)
            tf.summary.scalar('max',tf.reduce_max(var))
            tf.summary.scalar('min',tf.reduce_min(var))
            tf.summary.histogram('histogram',var)
    def hiden_layer(x,output_dim,scope='hidden_layer',act=tf.nn.relu,reuse=None):
        input_dim=x.get_shape().as_list()[-1]
        with tf.name_scope(scope):
            with tf.name_scope('weight'):

if __name__ == '__main__':
    # lookdata()
    full_connected()
    # open()
#
