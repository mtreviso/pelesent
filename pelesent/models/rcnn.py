import numpy as np
import logging
from pprint import pformat

from keras.models import Model, Sequential
from keras.layers import *

from pelesent.models.nn import NeuralNetwork

logger = logging.getLogger(__name__)

class RCNN(NeuralNetwork):

	def build(self, nb_filter=100, filter_length=7, stride=1, pool_length=3, cnn_activation='relu', 
					nb_hidden=200, rnn='LSTM', rnn_activation='sigmoid', dropout_rate=0.5, verbose=True):
		
		logger.info('Building...')
		inputs = []
		padding = filter_length // 2
		pool_padding = pool_length // 2

		RNN = LSTM if rnn == 'LSTM' else GRU
		
		sequence 		= Input(name='input_source', shape=(self.input_length, ), dtype='int32')
		embedded 		= Embedding(self.emb_vocab_size, self.emb_size, input_length=self.input_length, weights=[self.emb_weights])(sequence)
		
		drop 			= Dropout(dropout_rate)(embedded)
		
		cnn1d_pad 		= ZeroPadding1D(padding)(drop)
		cnn1d 			= Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation=cnn_activation, 
										subsample_length=stride, border_mode='valid')(cnn1d_pad)
	
		maxpooling_pad 	= ZeroPadding1D(pool_padding)(cnn1d)
		maxpooling 		= MaxPooling1D(pool_length=pool_length, border_mode='valid', stride=stride)(maxpooling_pad)

		forward_rnn 	= RNN(nb_hidden, activation=rnn_activation)(maxpooling)
		output 			= Dense(output_dim=self.nb_classes, activation='softmax', name='output_source')(forward_rnn)
		
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
