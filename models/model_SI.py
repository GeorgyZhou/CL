#!/usr/bin/python
# Copyright (c) 2018 Michael Zhou
#
# This file is the code to of the Synaptic Intelligence model:
# Zenke, F., Poole, B., and Ganguli, S. (2017). Continual Learning Through
# Synaptic Intelligence. In Proceedings of the 34th International Conference on
# Machine Learning, D. Precup, and Y.W. Teh, eds. (International Convention
# Centre, Sydney, Australia: PMLR), pp. 3987-3995.
# http://proceedings.mlr.press/v70/zenke17a.html

import numpy as np

import tensorflow as tf

class SIModel:
  def __init__(self, *args, **kwargs):
    pass

  def fit(self, train_type, task_dict, *args, **kwargs):
    pass

  def evaluate(self, *args, **kwargs):
    pass

  def summary(self):
    pass

  def predict(self, *args, **kwargs):
    pass

  def save_weights(self, file_prefix, overwrite=True):
    pass

  def load_weights(self, file_prefix):
    pass



  def __get_values_list(self, key='omega'):
    """Returns list of numerical values such as for instance omegas in reproducible order.

    Args:
      key: key for values to be extracted.
    Returns:
      A list containing values for that key.
    """
    variables = self.vars[key]
    values = []
    for p in self.weights:
      value = K.get_value(tf.reshape(variables[p],(-1,)))
      values.append(value)
    return values

  @staticmethod
  def __quadratic_regularizer(weights, vars, norm=2):
    """Compute the regularization term.

    Args:
      weights: list of Variables
      vars: dict from variable name to dictionary containing the variables.
            Each set of variables is stored as a dictionary mapping from weights to variables.
            For example, vars['grads'][w] would retreive the 'grads' variable for weight w
      norm: power for the norm of the (weights - consolidated weight)
    Returns:
      scalar Tensor regularization term
    """
    reg = 0.0
    for w in weights:
      reg += tf.reduce_sum(vars['omega'][w] * (w - vars['cweights'][]))
    return reg

CLModel = SIModel