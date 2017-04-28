import numpy as np
import logging
from pprint import pformat

from keras.models import Model, Sequential
from keras.layers import *

from pelesent.models.nn import NeuralNetwork

logger = logging.getLogger(__name__)

class RCNN(NeuralNetwork):

	def build(self, nb_filter=200, filter_length=7, stride=1, pool_length=3, cnn_activation='relu', 
					nb_hidden=100, rnn='LSTM', rnn_activation='tanh', dropout_rate=0.5, verbose=True):
		
		logger.info('Building...')
		inputs = []
		padding = filter_length // 2
		pool_padding = pool_length // 2

		RNN = LSTM if rnn == 'LSTM' else GRU
		
		sequence 		= Input(name='input_source', shape=(self.input_length, ), dtype='int32')
		embedded 		= Embedding(self.emb_vocab_size, self.emb_size, input_length=self.input_length, weights=[self.emb_weights])(sequence)
		
		cnn1d 			= Conv1D(nb_filter, filter_length, activation=cnn_activation)(embedded)
		maxpooling 		= MaxPooling1D(pool_length)(cnn1d)
		
		forward_rnn 	= RNN(nb_hidden, activation=rnn_activation)(maxpooling)
		backward_rnn 	= RNN(nb_hidden, go_backwards=True, activation=rnn_activation)(maxpooling)
		merge_rnn 		= Add()([forward_rnn, backward_rnn])
		drop 			= Dropout(dropout_rate)(merge_rnn)
		output 			= Dense(self.nb_classes, activation='softmax', name='output_source')(drop)
		
		self.classifier = Model(inputs=[sequence], outputs=output)

		logger.info('Compiling...')
		self._compile()
		if verbose:
			self._summary()
	
	def _compile(self):
		self.classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy')

	def _summary(self):
		self.classifier.summary()
		logger.debug('Model built: {}'.format(pformat(self.classifier.get_config())))
