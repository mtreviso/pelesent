import numpy as np
import logging
from pprint import pformat

from keras.models import Model
from keras.layers import *

from pelesent.models import NeuralNetwork

logger = logging.getLogger(__name__)

class CNN(NeuralNetwork):

	def build(self, nb_filter=100, filter_length=3, stride=1, pool_length=3, cnn_activation='relu', 
					nb_hidden=200, rnn='LSTM', rnn_activation='sigmoid', dropout_rate=0.5, verbose=True):
		
		logger.info('Building...')
		inputs = []
		
		sequence 		= Input(name='input_source', shape=(self.input_length, ), dtype='int32')
		embedded 		= Embedding(self.emb_vocab_size, self.emb_size, input_length=self.input_length, weights=[self.emb_weights])(sequence)
		drop 			= Dropout(0.2)(embedded)
		cnn1d 			= Convolution1D(nb_filter=200, filter_length=3, activation='relu', subsample_length=stride)(drop)
		maxp 			= GlobalMaxPooling1D()(cnn1d)
		dense 			= Dense(200, activation='relu')(maxp)
		drop2 			= Dropout(0.2)(dense)
		output 			= Dense(output_dim=self.nb_classes, activation='softmax', name='output_source')(drop2)

		self.classifier = Model(input=[sequence], output=output)
		logger.info('Compiling...')
		self._compile()
		if verbose:
			self._summary()
	
	def _compile(self):
		self.classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy')

	def _summary(self):
		self.classifier.summary()
		logger.debug('Model built: {}'.format(pformat(self.classifier.get_config())))
