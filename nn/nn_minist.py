import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist=input_data.read_data_sets("../tmp/data",one_hot=True)
pic_size=28
input_node=pic_size**2
classes=10
layer1=150
layer2=50
batch=100
lr=10e-3
data_size=len(mnist.validation.labels)
print(data_size)   #5000个数据

##############data
x=tf.placeholder(tf.float32,shape=(None,input_node),name='x-in')
y=tf.placeholder(tf.float32,shape=(None,classes),name='y')


#############inference


###定义前向传播过程
with tf.variable_scope('layer1'):
    w1 = tf.Variable(tf.random_normal([input_node, layer1], stddev=1, seed=1), name='w1')  # 数据可能一个一个输入的
    b1 = tf.Variable(tf.random_normal([layer1], stddev=1, seed=1), name='b1')

with tf.variable_scope('layer2'):
    w2 = tf.Variable(tf.random_normal([layer1, layer2], stddev=1, seed=1), name='w2')
    b2 = tf.Variable(tf.random_normal([layer2], stddev=1, seed=1), name='b2')
with tf.variable_scope('layer3'):
    w3 = tf.Variable(tf.random_normal([layer2, classes], stddev=1, seed=1), name='w3')  # 数据可能一个一个输入的
    b3 = tf.Variable(tf.random_normal([classes], stddev=1, seed=1), name='b3')

def two_network(nn_input):
    with tf.variable_scope('two_network'):
        s1 = tf.sigmoid((tf.matmul(x, w1)+b1))
        s2= tf.sigmoid(tf.matmul(s1, w2)+ b2)
        return tf.sigmoid(tf.matmul(s2, w3)+b3)
net=two_network(x)
y_pre=two_network(x)

# loss=-tf.reduce_mean(tf.add(tf.matmul(y,tf.log(y_pre)),tf.matmul((1-y),tf.log(1-y_pre))) )  ###
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_pre, 1))
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr, name='optimizer')
train_op=optimizer.minimize(loss)

################train
sess=tf.Session()
sess.run(tf.global_variables_initializer())
for e in range(5000):
# datafeed={x:mnist.validation.images,y:mnist.validation.labels}
    start = (e * batch) % data_size
    end = (start + batch+1) % data_size
    sess.run(train_op,feed_dict={x:mnist.validation.images[start:end],y:mnist.validation.labels[start:end]})
    # if e%500==0:

sess.close()
