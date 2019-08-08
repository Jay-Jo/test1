import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
with open('logistic_regression_data.txt', 'r') as f:
    data_list = [i.split('\n')[0].split('\t') for i in f.readlines()]
    data = [(float(i[0]), float(i[1]), float(i[2])) for i in data_list]
#标准化
x0_max = max([i[0] for i in data])
x1_max = max([i[1] for i in data])
data = [(i[0]/x0_max, i[1]/x1_max, i[2]) for i in data]

x0 = list(filter(lambda x: x[-1] == 0.0, data)) # 选择第一类的点
x1 = list(filter(lambda x: x[-1] == 1.0, data)) # 选择第二类的点

plot_x0 = [i[0] for i in x0]
plot_y0 = [i[1] for i in x0]
plot_x1 = [i[0] for i in x1]
plot_y1 = [i[1] for i in x1]


plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bo', label='x_1')
plt.legend(loc='best')

###数据处理完了

lr=0.01
np_data = np.array(data, dtype='float32') # 转换成 numpy array
x= tf.constant(np_data[:, 0:2], name='x') # 转换成 Tensor, 大小是 [100, 2]
print((np_data[:, -1]).shape)
y = tf.expand_dims(tf.constant(np_data[:, -1]), axis=-1) # 转换成 Tensor，大小是 [100, 1]
# y = tf.constant(np_data[:, -1])# 转换成 Tensor，大小是 [100, 1]



w=tf.Variable(initial_value=tf.random_normal(shape=[2,1],seed=2017),dtype=tf.float32,name='weight')
b=tf.Variable(initial_value=0,dtype=tf.float32,name='bias')


with tf.variable_scope('logstic'):
    y_pre=tf.sigmoid(tf.matmul(x,w)+b)

loss=-tf.reduce_mean(y*tf.log(y_pre)+(1-y)*tf.log(1-y_pre))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr, name='optimizer')
train_op=optimizer.minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for e in range(5000):
    sess.run(train_op)
    if e%50==0:
        y_true_label = y.eval(session=sess)
        y_pred_numpy = y_pre.eval(session=sess)
        y_pred_label = np.greater_equal(y_pred_numpy, 0.5).astype(np.float32)
        accuracy = np.mean(y_pred_label == y_true_label)
        loss_numpy = loss.eval(session=sess)
        print('Epoch %d, Loss: %.4f, Acc: %.4f' % (e + 1, loss_numpy, accuracy))

# # 画出最终分类效果
w_numpy = w.eval(session=sess)
b_numpy = b.eval(session=sess)

w0 = w_numpy[0]
w1 = w_numpy[1]
b0 = b_numpy

plot_x = np.arange(-3, 3, 0.01)
plot_y = (-w0 * plot_x - b0) / w1

plt.plot(plot_x, plot_y, 'g', label='cutting line')
plt.plot(plot_x0, plot_y0, 'ro', label='x_0')
plt.plot(plot_x1, plot_y1, 'bo', label='x_1')
plt.legend(loc='best')
#
sess.close()
plt.show()
