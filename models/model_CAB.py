import tensorflow as tf
import numpy as np

from utils.cab_utils import sigma
from utils.cab_utils import sigma_prime


def NOT(C, out_mode="simple"):
  """
  Compute NOT operation of conceptor.

  @param R: conceptor matrix
  @param out_mode: output mode ("simple"/"complete")

  @return not_C: NOT C
  @return U: eigen vectors of not_C
  @return S: eigen values of not_C
  """
  
  dim = C.shape[0]
  
  not_C = np.eye(dim) - C
  
  if out_mode == "complete":
    U, S, _ = np.linalg.svd(not_C)
    return not_C, U, S
  else:
    return not_C


def AND(C, B, out_mode="simple", tol=1e-14):
  """
  Compute AND Operation of two conceptor matrices

  @param C: a conceptor matrix
  @param B: another conceptor matrix
  @param out_mode: output mode ("simple"/"complete")
  @param tol: adjust parameter for almost zero

  @return C_and_B: C AND B
  @return U: eigen vectors of C_and_B
  @return S: eigen values of C_and_B
  """
  
  dim = C.shape[0]
  
  UC, SC, _ = np.linalg.svd(C)
  UB, SB, _ = np.linalg.svd(B)
  
  num_rank_C = np.sum((SC > tol).astype(int))
  num_rank_B = np.sum((SB > tol).astype(int))
  
  UC0 = UC[:, num_rank_C:]
  UB0 = UB[:, num_rank_B:]
  
  W, sigma, _ = np.linalg.svd(UC0.dot(UC0.T) + UB0.dot(UB0.T))
  num_rank_sigma = np.sum((sigma > tol).astype(int))
  Wgk = W[:, num_rank_sigma:]
  
  C_and_B = Wgk.dot(np.linalg.inv(Wgk.T.dot(
    np.linalg.pinv(C, tol) + np.linalg.pinv(B, tol) - np.eye(dim)).dot(
    Wgk))).dot(Wgk.T)
  
  if out_mode == "complete":
    U, S, _ = np.linalg.svd(C_and_B)
    return C_and_B, U, S
  else:
    return C_and_B


def OR(R, Q, out_mode="simple"):
  """
  Compute OR operation of two conceptor matrices

  @param R: a conceptor matrix
  @param Q: another conceptor matrix
  @param out_mode: output mode ("simple"/"complete")

  @return R_or_Q: R OR Q
  @return U: eigen vectors of R_or_Q
  @return S: eigen values of R_or_Q
  """
  
  R_or_Q = NOT(AND(NOT(R), NOT(Q)))
  
  if out_mode == "complete":
    U, S, _ = np.linalg.svd(R_or_Q)
    return R_or_Q, U, S
  else:
    return R_or_Q


def PHI(C, gamma):
  """
  aperture adaptation of conceptor C by factor gamma

  @param C: conceptor matrix
  @param gamma: adaptation parameter, 0 <= gamma <= Inf

  @return C_new: updated new conceptor matrix
  """
  
  dim = C.shape[0]
  
  if gamma == 0:
    U, S, _ = np.linalg.svd(C)
    S[S < 1] = np.zeros((np.sum((S < 1).astype(float)), 1))
    C_new = U.dot(S).dot(U.T)
  elif gamma == np.Inf:
    U, S, _ = np.linalg.svd(C)
    S[S > 0] = np.zeros((np.sum((S > 0).astype(float)), 1))
    C_new = U.dot(S).dot(U.T)
  else:
    C_new = C.dot(np.linalg.inv(C + gamma ** -2 * (np.eye(dim) - C)))
  
  return C_new

class CABModel:

  def __init__(self, *args, **kwargs):
    self.input_shape = int(np.prod(kwargs.get('input_shape', (28, 28, 1))))
    self.output_shape = kwargs.get('output_shape', 10)
    self.n_hidden_units = 100
    
    print (self.input_shape)

    self.a_0 = tf.placeholder(tf.float32, [None, self.input_shape])
    self.y = tf.placeholder(tf.float32, [None, self.output_shape])

    w_1 = tf.Variable(tf.truncated_normal([self.input_shape + 1, self.n_hidden_units], stddev=0.1))
    w_2 = tf.Variable(tf.truncated_normal([self.n_hidden_units + 1, self.output_shape], stddev=0.1))

    w_old_1 = tf.Variable(tf.zeros([self.input_shape + 1, self.n_hidden_units]))
    w_old_2 = tf.Variable(tf.zeros([self.n_hidden_units + 1, 10]))

    # Conceptors for used spaces
    self.A_0 = np.zeros([self.input_shape + 1, self.input_shape + 1])
    self.A_1 = np.zeros([self.n_hidden_units + 1, self.n_hidden_units + 1])

    # Conceptors for free spaces
    self.F_0 = tf.Variable(tf.eye(self.input_shape + 1))
    self.F_1 = tf.Variable(tf.eye(self.n_hidden_units + 1))

    # Forward Pass, ab_i is the state vector together with bias
    self.ab_0 = tf.concat([self.a_0, tf.tile(tf.ones([1, 1]), [tf.shape(self.a_0)[0], 1])], 1)
    z_1 = tf.matmul(self.ab_0, w_1)
    a_1 = sigma(z_1)
    self.ab_1 = tf.concat([a_1, tf.tile(tf.ones([1, 1]), [tf.shape(a_1)[0], 1])], 1)
    z_2 = tf.matmul(self.ab_1, w_2)
    a_2 = sigma(z_2)

    diff = tf.subtract(a_2, self.y)

    # Backward Pass
    reg2 = tf.Variable(0.001)
    reg1 = tf.Variable(0.001)

    d_z_2 = tf.multiply(diff, sigma_prime(z_2))
    d_w_2 = tf.matmul(tf.transpose(tf.matmul(self.ab_1, self.F_1)), d_z_2)

    inc_w_2 = tf.subtract(w_2, w_old_2)
    reg_w_2 = tf.multiply(reg2, inc_w_2)
    d_w_2 = tf.add(d_w_2, reg_w_2)

    d_ab_1 = tf.matmul(d_z_2, tf.transpose(w_2))
    d_a_1 = d_ab_1[:, :-1]
    d_z_1 = tf.multiply(d_a_1, sigma_prime(z_1))
    d_w_1 = tf.matmul(tf.transpose(tf.matmul(self.ab_0, self.F_0)), d_z_1)

    inc_w_1 = tf.subtract(w_1, w_old_1)
    reg_w_1 = tf.multiply(reg1, inc_w_1)
    d_w_1 = tf.add(d_w_1, reg_w_1)

    eta = tf.constant(0.1)
    self.step = [
      tf.assign(w_1, tf.subtract(w_1, tf.multiply(eta, d_w_1))),
      tf.assign(w_2, tf.subtract(w_2, tf.multiply(eta, d_w_2)))
    ]

    # Compute Classification Accuracy
    acct_mat = tf.equal(tf.argmax(a_2, 1), tf.argmax(self.y, 1))
    self.acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))
    self.loss_res = tf.nn.softmax(a_2)

    # Update the old weights, which are the weights before training a task
    updateW_old = [tf.assign(w_old_1, w_1), tf.assign(w_old_2, w_2)]

    self.sess = tf.InteractiveSession()
    self.sess.run(tf.global_variables_initializer())

  def fit(self, traintype, task_dict, *args, **kwargs):
    for i in range(100):
      self.sess.run(self.step, feed_dict={
        self.a_0: kwargs['x'].reshape((len(kwargs['x']), self.input_shape)),
        self.y: kwargs['y']
      })
    return None

  def evaluate(self, *args, **kwargs):
    res = self.sess.run([self.loss_res, self.acct_res], feed_dict={
        self.a_0: kwargs['x'].reshape((len(kwargs['x']), self.input_shape)),
        self.y: kwargs['y']
    })
    print(res)
    return res

  def summary(self):
    pass

  def predict(self, *args, **kwargs):
    pass

  def save_weights(self, fileprefix, overwrite=True):
    
    pass

  def load_weights(self, fileprefix):
    pass

  def __pre_fit(self, xs):
    ab0_collection = self.sess.run(self.ab_0, feed_dict={self.a_0: xs})
  
    alpha = 4
    R_a0 = (ab0_collection.T).dot(ab0_collection) / ab0_collection.shape[0]
    U_a0, S_a0, _ = np.linalg.svd(R_a0)
    S_C0 = (np.diag(S_a0).dot(np.linalg.inv(
      np.diag(S_a0) + alpha ** (-2) * np.eye(ab0_collection.shape[1]))))
    S0 = np.diag(S_C0)
    C0 = U_a0.dot(np.diag(S0)).dot(U_a0.T)
  
    # Collecting activation vectors to compute conceptors on the hidden layer
    ab1_collection = self.sess.run(self.ab_1, feed_dict={self.a_0: xs})
  
    alpha1 = 4
    R_a1 = (ab1_collection.T).dot(ab1_collection) / ab1_collection.shape[0]
    U_a1, S_a1, _ = np.linalg.svd(R_a1)
    S_C1 = (np.diag(S_a1).dot(np.linalg.inv(
      np.diag(S_a1) + alpha1 ** (-2) * np.eye(ab1_collection.shape[1]))))
    S1 = np.diag(S_C1)
    C1 = U_a1.dot(np.diag(S1)).dot(U_a1.T)
  
    # Update the conceptors for used spaces on each layer
    self.A_0 = OR(C0, self.A_0)
    self.A_1 = OR(C1, self.A_1)
  
    # Update the conceptors for free space on each layer
    F0 = NOT(self.A_0)
    F1 = NOT(self.A_1)
  
    updateF = [tf.assign(self.F_0, tf.cast(F0, tf.float32)),
               tf.assign(self.F_1, tf.cast(F1, tf.float32))]
    self.sess.run(updateF)

CLModel = CABModel