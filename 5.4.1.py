import  tensorflow as tf

# v1 = tf.Variable(tf.constant(1.0,shape=[1], name='v1'))
# v2 = tf.Variable(tf.constant(2.0,shape=[1], name='v2'))
#
# result = v1 + v2
#
# init_op = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     # sess.run(init_op)
#     # saver.save(sess, '/path/to/model/model.ckpt')
#     saver.restore(sess, '/path/to/model/model.ckpt')
#     print(sess.run(result))
# saver = tf. train.import_meta_graph('/path/to/model/model.ckpt.meta')
# with tf.Session() as sess:
#     saver.restore(sess, '/path/to/model/model.ckpt')
#     print(sess.run(tf.get_default_graph().get_tensor_by_name('add:0')))
v = tf.Variable(0,dtype=tf.float32, name='v')
for variables in tf.global_variables():
    print(variables.name)
ema = tf.train.ExponentialMovingAverage(0.99)
maintain_averages_op = ema.apply(tf.global_variables())
for variables in tf.global_variables():
    print(variables.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(tf.assign(v,10))
    sess.run(maintain_averages_op)
    saver.save(sess, '/path/to/model/model.ckpt')
    print(sess.run([v,ema.average(v)]))
