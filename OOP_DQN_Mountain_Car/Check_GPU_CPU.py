# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

# import tensorflow as tf
# with tf.device('/gpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#     c = tf.matmul(a, b)
#
# with tf.Session() as sess:
#     print (sess.run(c))

import tensorflow as tf
import keras
config = tf.ConfigProto(device_count = {'GPU': 2 , 'CPU': 1} )
sess = tf.Session(config=config)
keras.backend.set_session(sess)