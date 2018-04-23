import tensorflow as tf
import numpy as np

from utils.cab_utils import sigma
from utils.cab_utils import sigma_prime



class CABModel:

  def __init__(self, *args, **kwargs):
    self.input_shape = int(np.prod(kwargs.get('input_shape', (28, 28, 1))))
    self.output_shape = kwargs.get('output_shape', 10)
    self.n_hidden_units = 100

    a_0 = tf.placeholder(tf.float32, [None, self.input_shape])
    y = tf.placeholder(tf.float32, [None, self.output_shape])

    w_1 = tf.Variable(tf.truncated_normal([self.input_shape, self.n_hidden_units], stddev=0.1))
    w_2 = tf.Variable(tf.truncated_normal([self.n_hidden_units + 1, self.input_shape], stddev=0.1))

    w_old_1 = tf.Variable(tf.zeros([self.input_shape, self.input_shape]))
    w_old_2 = tf.Variable(tf.zeros([self.n_hidden_units + 1, 10]))

    # Conceptors for used spaces
    A_0 = np.zeros([self.input_shape, self.input_shape])
    A_1 = np.zeros([self.n_hidden_units + 1, self.n_hidden_units])

    # Conceptors for free spaces
    F_0 = tf.Variable(tf.eye(self.input_shape))
    F_1 = tf.Variable(tf.eye(self.n_hidden_units + 1))

    # Forward Pass, ab_i is the state vector together with bias
    ab_0 = tf.concat([a_0, tf.tile(tf.ones([1, 1]), [tf.shape(a_0)[0], 1])], 1)
    z_1 = tf.matmul(ab_0, w_1)
    a_1 = sigma(z_1)
    ab_1 = tf.concat([a_1, tf.tile(tf.ones([1, 1]), [tf.shape(a_1)[0], 1])], 1)
    z_2 = tf.matmul(ab_1, w_2)
    a_2 = sigma(z_2)

    diff = tf.subtract(a_2, y)

    # Backward Pass
    reg2 = tf.Variable(0.001)
    reg1 = tf.Variable(0.001)

    d_z_2 = tf.multiply(diff, sigma_prime(z_2))
    d_w_2 = tf.matmul(tf.transpose(tf.matmul(ab_1, F_1)), d_z_2)

    inc_w_2 = tf.subtract(w_2, w_old_2)
    reg_w_2 = tf.multiply(reg2, inc_w_2)
    d_w_2 = tf.add(d_w_2, reg_w_2)

    d_ab_1 = tf.matmul(d_z_2, tf.transpose(w_2))
    d_a_1 = d_ab_1[:, :-1]
    d_z_1 = tf.multiply(d_a_1, sigma_prime(z_1))
    d_w_1 = tf.matmul(tf.transpose(tf.matmul(ab_0, F_0)), d_z_1)

    inc_w_1 = tf.subtract(w_1, w_old_1)
    reg_w_1 = tf.multiply(reg1, inc_w_1)
    d_w_1 = tf.add(d_w_1, reg_w_1)

    eta = tf.constant(0.1)
    self.step = [
      tf.assign(w_1,
                tf.subtract(w_1, tf.multiply(eta, d_w_1)))

      , tf.assign(w_2,
                  tf.subtract(w_2, tf.multiply(eta, d_w_2)))
    ]

    # Compute Classification Accuracy
    acct_mat = tf.equal(tf.argmax(a_2, 1), tf.argmax(y, 1))
    self.acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

    # Update the old weights, which are the weights before training a task
    updateW_old = [tf.assign(w_old_1, w_1), tf.assign(w_old_2, w_2)]

    self.sess = tf.InteractiveSession()
    self.sess.run(tf.global_variables_initializer())

  def fit(self, traintype, task_dict, *args, **kwargs):
    print(kwargs['x'].shape)
    # for i in xrange(10000):
      # res = self.sess.run(self.acct_res, feed_dict={a_0: , })
    pass

  def evaluate(self):
    pass

  def summary(self):
    pass

  def predict(self, *args, **kwargs):
    pass

  def save_weights(self, fileprefix, overwrite=True):
    pass

  def load_weights(self, fileprefix):
    pass


CLModel = CABModel