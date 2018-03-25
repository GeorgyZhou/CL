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

def _ema(decay, prev_val, new_val):
    """Compute exponential moving average.

    Args:
        decay: 'sum' to sum up values, otherwise decay in [0, 1]
        prev_val: previous value of accumulator
        new_val: new value
    Returns:
        updated accumulator
    """
    if decay == 'sum':
        return prev_val + new_val
    assert isinstance(decay, float)
    return decay * prev_val + (1.0 - decay) * new_val

def _compute_updates(opt, loss, weights):
  update_ops = opt.get_updates(weights, [], loss)
  deltas, new_update_op = SIModel._extract_weight_changes(weights, update_ops)
  grads = tf.gradients(loss, weights)
  # Make sure  that deltas are computed _before_ the weight is updated
  return new_update_op, grads, deltas


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
    reg += tf.reduce_sum(vars['omega'][w] * (w - vars['cweights'][w]))
  return reg


_path_int_protocol = lambda omega_decay, xi: (
   'path_int[omega_decay=%s,xi=%s]'%(omega_decay,xi),
    {
        'init_updates':  [
            ('cweights', lambda vars, w, prev_val: w.value() ),
            ],
        'step_updates':  [
            ('grads2', lambda vars, w, prev_val: prev_val -vars['unreg_grads'][w] * vars['deltas'][w] ),
            ],
        'task_updates':  [
            ('omega', lambda vars, w, prev_val: tf.nn.relu(_ema(omega_decay, prev_val, vars['grads2'][w]/((vars['cweights'][w]-w.value())**2+xi)))),
            #('cached_grads2', lambda vars, w, prev_val: vars['grads2'][w]),
            #('cached_cweights', lambda vars, w, prev_val: vars['cweights'][w]),
            ('cweights',  lambda opt, w, prev_val: w.value()),
            ('grads2', lambda vars, w, prev_val: prev_val*0.0 ),
        ],
        'regularizer_fn': _quadratic_regularizer,
    })

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

  def _get_updates(self, weights):
    self.weights = weights
    self.regularizer = _quadratic_regularizer(weights, self.vars)

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





CLModel = SIModel