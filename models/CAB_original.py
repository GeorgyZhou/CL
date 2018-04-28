import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from IPython import display
from copy import deepcopy
from tensorflow.examples.tutorials.mnist import input_data

'''
Created on May 25, 2015
Modified on Feb 05, 2018
@author: Xu He
@note: Logical operations on conceptors
'''

import numpy as np;

def NOT(C, out_mode = "simple"):
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
  
def AND(C, B, out_mode = "simple", tol = 1e-14):
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
  
  C_and_B = Wgk.dot(np.linalg.inv(Wgk.T.dot(np.linalg.pinv(C, tol) + np.linalg.pinv(B, tol) - np.eye(dim)).dot(Wgk))).dot(Wgk.T)
  

  if out_mode =="complete":
    U, S, _ = np.linalg.svd(C_and_B)
    return C_and_B, U, S
  else:
    return C_and_B
  

def OR(R, Q, out_mode = "simple"):
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


#Activation function and its derivative
def sigma(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.negative(x))))

def sigmaprime(x):
    return tf.multiply(sigma(x), tf.subtract(tf.constant(1.0), sigma(x)))


#Create 10 permuted MNIST datasets
def permute_mnist(mnist):
    perm_inds = range(mnist.train.images.shape[1])
    np.random.shuffle(perm_inds)
    mnist2 = deepcopy(mnist)
    sets = ["train", "validation", "test"]
    for set_name in sets:
        this_set = getattr(mnist2, set_name) # shallow copy
        this_set._images = np.transpose(np.array([this_set.images[:,c] for c in perm_inds]))
    return mnist2

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist_list = [mnist]
for j in xrange(9):
    mnist_list.append(permute_mnist(mnist))

#Define a 2 layer feedfoward network with 100 hidden neurons 
a_0 = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

middle = 100

w_1 = tf.Variable(tf.truncated_normal([785, middle], stddev=0.1))
w_2 = tf.Variable(tf.truncated_normal([middle+1, 10], stddev=0.1))

w_old_1 = tf.Variable(tf.zeros([785, middle]))
w_old_2 = tf.Variable(tf.zeros([middle+1, 10]))



#Conceptors for used spaces
A_0 = np.zeros([785, 785])
A_1 = np.zeros([middle+1, middle+1])

#Conceptors for free spaces
F_0 = tf.Variable(tf.eye(785))
F_1 = tf.Variable(tf.eye(middle+1))


#Forward Pass, ab_i is the state vector together with bias
ab_0 = tf.concat([a_0, tf.tile(tf.ones([1,1]), [tf.shape(a_0)[0], 1])], 1)
z_1 = tf.matmul(ab_0, w_1)
a_1 = sigma(z_1)
ab_1 = tf.concat([a_1, tf.tile(tf.ones([1,1]), [tf.shape(a_1)[0], 1])], 1)
z_2 = tf.matmul(ab_1, w_2)
a_2 = sigma(z_2)

diff = tf.subtract(a_2, y)


#Backward Pass
reg2 = tf.Variable(0.001)
reg1 = tf.Variable(0.001)

d_z_2 = tf.multiply(diff, sigmaprime(z_2))
d_w_2 = tf.matmul(tf.transpose(tf.matmul(ab_1,F_1)), d_z_2)

inc_w_2 = tf.subtract(w_2, w_old_2)
reg_w_2 = tf.multiply(reg2, inc_w_2)
d_w_2 = tf.add(d_w_2, reg_w_2)


d_ab_1 = tf.matmul(d_z_2, tf.transpose(w_2))
d_a_1 = d_ab_1[:, :-1]
d_z_1 = tf.multiply(d_a_1, sigmaprime(z_1))
d_w_1 = tf.matmul(tf.transpose(tf.matmul(ab_0,F_0)), d_z_1)

inc_w_1 = tf.subtract(w_1, w_old_1)
reg_w_1 = tf.multiply(reg1, inc_w_1)
d_w_1 = tf.add(d_w_1, reg_w_1)

eta = tf.constant(0.1)
step = [
    tf.assign(w_1,
            tf.subtract(w_1, tf.multiply(eta, d_w_1)))

  , tf.assign(w_2,
            tf.subtract(w_2, tf.multiply(eta, d_w_2)))
]

#Compute Classification Accuracy
acct_mat = tf.equal(tf.argmax(a_2, 1), tf.argmax(y, 1))
acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

#Update the old weights, which are the weights before training a task
updateW_old = [tf.assign(w_old_1, w_1), tf.assign(w_old_2, w_2)]


#Initialize variables
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#Training the network sequentially on the 10 datasets
task_num = 10
avg_list = []
SA0_list = []
SA1_list = []
prev_list = [[] for x in xrange(task_num)]
     

#Iterate over 10 tasks
for j in xrange(task_num):
    print "Training MNIST %d" % (j+1)
    
    #Update the parameters for 10000 times
    for i in xrange(10000):
        batch_xs, batch_ys = mnist_list[j].train.next_batch(50)

        sess.run(step, feed_dict = {a_0: batch_xs, y: batch_ys})
        
        #Print validation accuracy every 1000 steps
        if i % 1000 == 0:
            res = sess.run(acct_res, feed_dict =
                           {a_0:mnist_list[j].validation.images[:1000],
                            y : mnist_list[j].validation.labels[:1000]})
            print "Validation accuracy:", res/1000

    sess.run(updateW_old)
    
    print "w_1 norm", sess.run(tf.reduce_sum(tf.norm(w_1)))
    print "w_2 norm", sess.run(tf.reduce_sum(tf.norm(w_2)))
    
    #Print the accuracies on testing set of the task just trained on
    res = sess.run(acct_res, feed_dict =
                           {a_0: mnist_list[j].test.images[:100000],
                            y : mnist_list[j].test.labels[:100000]})
    print "Accuracy on Current Dataset", res/mnist.test.labels.shape[0]
    
    res_sum = 0
    
    print "Test on all Previous Datasets:" 
    for i in xrange(j+1):
        res = sess.run(acct_res, feed_dict =
                               {a_0: mnist_list[i].test.images[:100000],
                                y : mnist_list[i].test.labels[:100000]})
        acc_res = res/mnist.test.labels.shape[0]
        print acc_res
        prev_list[i].append(acc_res)
        res_sum += acc_res
        avg_res = res_sum/(j+1)
    print "Current Average Accuracy:", avg_res

    avg_list.append(avg_res)
    
    #Collecting activation vectors to compute conceptors on the input layer
    batch_xs, batch_ys = mnist_list[j].train.next_batch(500)
    a0_collection = batch_xs
    ab0_collection = sess.run(ab_0, feed_dict = {a_0: a0_collection})

    alpha = 4
    R_a0 = (ab0_collection.T).dot(ab0_collection) / ab0_collection.shape[0]
    U_a0, S_a0, _ = np.linalg.svd(R_a0)
    S_C0 = (np.diag(S_a0).dot(np.linalg.inv(np.diag(S_a0) + alpha ** (-2) * np.eye(ab0_collection.shape[1]))))
    S0 = np.diag(S_C0)    
    C0 = U_a0.dot(np.diag(S0)).dot(U_a0.T)

    #Collecting activation vectors to compute conceptors on the hidden layer
    ab1_collection = sess.run(ab_1, feed_dict = {a_0: a0_collection})
    
    alpha1 = 4
    R_a1 = (ab1_collection.T).dot(ab1_collection) / ab1_collection.shape[0]
    U_a1, S_a1, _ = np.linalg.svd(R_a1)
    S_C1 = (np.diag(S_a1).dot(np.linalg.inv(np.diag(S_a1) + alpha1 ** (-2) * np.eye(ab1_collection.shape[1]))))
    S1 = np.diag(S_C1)
    C1 = U_a1.dot(np.diag(S1)).dot(U_a1.T)

    #Update the conceptors for used spaces on each layer
    A_0 = OR(C0, A_0)
    A_1 = OR(C1, A_1)

    #Update the conceptors for free space on each layer
    F0 = NOT(A_0)
    F1 = NOT(A_1)
    
    updateF = [tf.assign(F_0, tf.cast(F0, tf.float32)), tf.assign(F_1, tf.cast(F1, tf.float32))]
    sess.run(updateF)
