import os
import tensorflow as tf

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Computational Graph - FLOW
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
addition = tf.add(X, Y, name="addition")

# Running the graph with data - TENSORS
with tf.Session() as session:
    result = session.run(addition, feed_dict={X: [1, 2, 10], Y: [4, 2, 10]})
    print(result)
