
import tensorflow as tf

x_placeholder = tf.placeholder(tf.int32, shape=[4,2])
y_placeholder = tf.placeholder(tf.int32, shape=[4])

Theta1 = tf.Variable(tf.random_normal([2,2]))
Theta2 = tf.Variable(tf.random_normal([2,1]))

Bias1 = tf.Variable(tf.ones([2]))
Bias2 = tf.Variable(tf.ones([1]))

A2 = tf.sigmoid(tf.matmul(tf.to_float(x_placeholder), Theta1) + Bias1)
Hypothesis = tf.sigmoid(tf.matmul(A2, Theta2) + Bias2)

cost = tf.reduce_sum(tf.to_float(y_placeholder) * tf.log(Hypothesis))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cost)

XOR_X = [[0,0],[0,1],[1,0],[1,1]]
XOR_Y = [0,1,1,0]

init = tf.initialize_all_variables()
sess = tf.Session()

sess.run(init)

for i in range(1000):
	sess.run(train_step, feed_dict={x_placeholder: XOR_X, y_placeholder: XOR_Y})


