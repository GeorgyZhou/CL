import tensorflow as tf

#Activation function and its derivative
def sigma(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))

def sigma_prime(x):
    return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))