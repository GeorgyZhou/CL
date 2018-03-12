import numpy as np
np.random.seed(123)

from keras.models import Sequential
from keras.layers import Dense, Reshape
from keras.regularizers import l1_l2, l2
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras import backend as KB

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

############################ Keras models ###########################

class NN:

	def __init__(self, *args, **kwargs):
		self.input_shape = kwargs['input_shape'] if 'input_shape' in kwargs else (28,28,1)
		self.output_shape = kwargs['output_shape'] if 'output_shape' in kwargs else 10
		self.internal_input_shape = (np.prod(self.input_shape),)

		self.model = Sequential()
		self.model.add(Reshape(target_shape=self.internal_input_shape, 
				input_shape=self.input_shape))
		self.model.add(Dense(24, activation='relu'))
		self.model.add(Dense(24, activation='relu'))
		self.model.add(Dense(self.output_shape, activation='softmax'))
		self.model.compile(loss='categorical_crossentropy', 
				optimizer=RMSprop(), metrics=['accuracy'])

	def fit(self, traintype, task_dict, *args, **kwargs):
		return self.model.fit(*args, **kwargs)

	def evaluate(self, *args, **kwargs):
		return self.model.evaluate(x=kwargs['x'], y=kwargs['y'], verbose=0)

	def summary(self):
		self.model.summary()

	def predict(self, *args, **kwargs):
		return self.model.predict(x=kwargs['x'], verbose=0)

	def save_weights(self, fileprefix, overwrite=True):
		self.model.save_weights(filepath=fileprefix+'.hdf5', overwrite=overwrite)

	def load_weights(self, fileprefix):
		self.model.load_weights(filepath=fileprefix+'.hdf5')

############################ Aliases ###########################

CLModel = NN