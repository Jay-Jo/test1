import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(2019)

def plot_decision_boundary(model, x, y):
    # 找到x, y的最大值和最小值, 并在周围填充一个像素
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    h = 0.01
    # 构建一个宽度为`h`的网格
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # 计算模型在网格上所有点的输出值
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # 画图显示
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), cmap=plt.cm.Spectral)

np.random.seed(1)
m = 400 # 样本数量
N = int(m/2) # 每一类的点的个数
D = 2 # 维度
x = np.zeros((m, D))
y = np.zeros((m, 1), dtype='uint8') # label 向量，0 表示红色，1 表示蓝色
a = 4

for j in range(2):
    ix = range(N*j,N*(j+1))
    t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
    r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
    x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j


plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), s=40, cmap=plt.cm.Spectral)


x = tf.constant(x, dtype=tf.float32, name='x')
y = tf.constant(y, dtype=tf.float32, name='y')
# 定义模型
w = tf.get_variable(initializer=tf.random_normal_initializer(), shape=(2, 1), dtype=tf.float32, name='weights')
b = tf.get_variable(initializer=tf.zeros_initializer(), shape=(1), dtype=tf.float32, name='bias')


def logistic_model(x):
    logit = tf.matmul(x, w) + b
    return tf.sigmoid(logit)

y_ = logistic_model(x)

# 构造训练
loss = tf.losses.log_loss(predictions=y_, labels=y)

lr = 1e-1
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss )
# 执行训练

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

for e in range(1000):
    sess.run(train_op)
    if (e + 1) % 100 == 0:
        loss_numpy = loss.eval(session=sess)
        print('Epoch %d: Loss: %.12f' % (e + 1, loss_numpy))

# model_input = tf.placeholder(shape=(None, 2), dtype=tf.float32, name='logistic_input')
# logistic_output = logistic_model(model_input)
plt.show()
#
# def plot_logistic(x_data):
#     y_pred_numpy = sess.run(logistic_output, feed_dict={model_input: x_data})
#     out = np.greater(y_pred_numpy, 0.5).astype(np.float32)
#     return np.squeeze(out)
# plot_decision_boundary(plot_logistic, x.eval(session=sess), y.eval(session=sess))