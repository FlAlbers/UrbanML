import tensorflow as tf

with tf.device('/device:GPU:0'):

# Create two random matrices

a = tf.random.normal([1000, 1000])

b = tf.random.normal([1000, 1000])

# Multiply the matrices

c = tf.matmul(a, b)

print(c)