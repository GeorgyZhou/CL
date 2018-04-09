import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.optimizers import Adam
from SI_original.pathint.optimizers import KOOptimizer, PATH_INT_PROTOCOL
from


class SIModel:
  def __init__(self, *args, **kwargs):
    self.keras_loss = kwargs.get('keras_loss', 'sgd')
    self.input_shape = kwargs.get('input_shape', (28,28,1))
    self.output_shape = kwargs.get('output_shape', 10)
    self.learning_rate = kwargs.get('learning_rate', 1e-3)
    assert isinstance(self.output_shape, int)
    assert isinstance(self.input_shape, tuple)
    assert isinstance(self.learning_rate, float)
    self.internal_input_shape = (np.prod(self.input_shape),)
    self.model = Sequential()
    self.model.add(Reshape(target_shape=self.internal_input_shape,
                           input_shape=self.input_shape))
    self.model.add(Dense(24, activation='relu'))
    self.model.add(Dense(24, activation='relu'))
    self.model.add(Dense(self.output_shape, activation='softmax'))
    protocol_name, protocol = PATH_INT_PROTOCOL(omega_decay='sum', xi=1e-3 )
    self.opt = KOOptimizer(Adam(lr=self.learning_rate, beta_1=0.9, beta_2=0.999),
                      model=self.model, **protocol)
    self.model.compile(loss='categorical_crossentropy',
                       optimizer=self.opt,
                       metrics=['accuracy'])

  def fit(self, traintype, task_dict, *args, **kwargs):
    history = self.model.fit(*args, **kwargs)
    if traintype == 'seq':
      self.opt.update_task_metrics(kwargs['x'], kwargs['y'], kwargs['batch_size'])
      self.opt.update_task_vars()
    return history


  def evaluate(self, *args, **kwargs):
    return self.model.evaluate(x=kwargs['x'], y=kwargs['y'], verbose=0)

  def summary(self):
    self.model.summary()

  def predict(self, *args, **kwargs):
    return self.model.predict(x=kwargs['x'], verbose=0)

  def save_weights(self, fileprefix, overwrite=True):
    self.model.save_weights(filepath=fileprefix + '.hdf5', overwrite=overwrite)

  def load_weights(self, fileprefix):
    self.model.load_weights(filepath=fileprefix + '.hdf5')

CLModel = SIModel