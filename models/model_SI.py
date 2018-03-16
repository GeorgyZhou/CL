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

  def _get_values_list(self, key='omega'):
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
  def _extract_weight_changes(weights, update_ops):
    """Given a list of weights and Assign ops, identify the change in weights.

    Args:
        weights: list of Variables
        update_ops: list of Assign ops, typically computed using Keras' opt.get_updates()

    Returns:
        list of Tensors containing the weight update for each variable
    """
    name_to_var = {v.name: v.value() for v in weights}
    weight_update_ops = list(filter(lambda x: x.op.inputs[0].name in name_to_var, update_ops))
    nonweight_update_ops = list(filter(lambda x: x.op.inputs[0].name not in name_to_var, update_ops))
    # Make sure that all the weight update ops are Assign ops
    for weight in weight_update_ops:
      if weight.op.type != 'Assign':
        raise ValueError('Update op for weight %s is not of type Assign.' % weight.op.inputs[0].name)
    weight_changes = [(new_w.op.inputs[1] - name_to_var[new_w.op.inputs[0].name]) for new_w, old_w in
                      zip(weight_update_ops, weights)]
    # Recreate the update ops, ensuring that we compute the weight changes before updating the weights
    with tf.control_dependencies(weight_changes):
      new_weight_update_ops = [tf.assign(new_w.op.inputs[0], new_w.op.inputs[1]) for new_w in weight_update_ops]
    return weight_changes, tf.group(*(nonweight_update_ops + new_weight_update_ops))

  @staticmethod
  def _compute_updates(opt, loss, weights):
    update_ops = opt.get_updates(weights, [], loss)
    deltas, new_update_op = SIModel._extract_weight_changes(weights, update_ops)
    grads = tf.gradients(loss, weights)
    # Make sure  that deltas are computed _before_ the weight is updated
    return new_update_op, grads, deltas

  @staticmethod
  def _quadratic_regularizer(weights, vars, norm=2):
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