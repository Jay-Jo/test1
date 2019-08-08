import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#原始数据
X=[ 1 ,2  ,3 ,4 ,5 ,6]
Y=[ 2.6 ,3.4 ,4.7 ,5.5 ,6.47 ,7.8]
plt.plot(X,Y,'bo')

l=len(X)

lr=0.05

x=tf.constant(np.array(X,dtype=np.float32),name='x')
y=tf.constant(np.array(Y,dtype=np.float32),name='y')

# a=tf.placeholer(tf.float32,shape=[1,l])
# b=tf.placeholer(tf.float32,shape=[1,l]

w=tf.Variable(initial_value=tf.random_normal(shape=(),seed=2017),dtype=tf.float32,name='weight')
b=tf.Variable(initial_value=0,dtype=tf.float32,name='bias')

with tf.variable_scope('linear'):  ##变量域
    y_pred=w*x+b

loss=tf.reduce_mean(tf.square(y-y_pred))
w_grad,b_grad=tf.gradients(loss,[w,b])
w_update=w.assign_sub(lr*w_grad)
b_update=b.assign_sub(lr*b_grad)

intil=tf.global_variables_initializer()
sess=tf.Session()
sess.run(intil)

for e in range(1000):
    sess.run([w_update,b_update])
    y_pred_numpy=y_pred.eval(session=sess)
    loss_numpy=loss.eval(session=sess)

    print('epoch:{},loss:{}'.format(e,loss_numpy))


sess.close()

plt.plot(X,Y,'bo',label='real')
plt.plot(X,y_pred_numpy,'ro',label='pre')
plt.legend()

plt.show()










# #用一次多项式拟合，相当于线性拟合
# z1 = np.polyfit(X, Y, 2)
# p1 = np.poly1d(z1)
# print (z1)  #[ 1.          1.49333333]
# print (p1)  # 1 x + 1.493
#
#
# import matplotlib.pyplot as plt
# import numpy as np
#
# x = np.arange(1,7)
# y = z1[0] * x*x + z1[1]*x+z1[2]
# plt.figure()
# plt.scatter(X, Y)
# plt.plot(x, y)
# plt.show()
