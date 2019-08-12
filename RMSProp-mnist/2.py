import tensorflow as tf
tf.add_to_collection('losses', tf.constant(2.2))
tf.add_to_collection('losses', tf.constant(3.))
with tf.Session() as sess:
    print(sess.run(tf.get_collection('losses')))
    print(sess.run(tf.add_n(tf.get_collection('losses'))))
