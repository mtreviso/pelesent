import numpy as np
import logging
from pprint import pformat

from keras.models import Model
from keras.layers import *

from pelesent.models import NeuralNetwork

logger = logging.getLogger(__name__)

class CRNN(NeuralNetwork):

	def build(self, nb_filter=100, filter_length=7, stride=1, pool_length=3, cnn_activation='relu', 
					nb_hidden=200, rnn='LSTM', rnn_activation='sigmoid', dropout_rate=0.5, verbose=True):
		
		logger.info('Building...')
		inputs = []
		RNN = LSTM if rnn == 'LSTM' else GRU
		
		sequence = Input(name='input_source', shape=(self.input_length,), dtype='int32')
		embedded = Embedding(self.emb_vocab_size, self.emb_size, input_length=self.input_length, weights=[self.emb_weights])(sequence)
		inputs.append(sequence)

		cnn1d_pad 		= ZeroPadding1D(padding)(embedded)
		cnn1d 			= Convolution1D(nb_filter=nb_filter, filter_length=filter_length, activation=cnn_activation, 
										subsample_length=stride, border_mode='valid')(cnn1d_pad)
	
		maxpooling_pad 	= ZeroPadding1D(pool_padding)(cnn1d)
		maxpooling 		= MaxPooling1D(pool_length=pool_length, border_mode='valid', stride=stride)(maxpooling_pad)

		forward_rnn 	= RNN(nb_hidden, return_sequences=True, activation=rnn_activation)(maxpooling)
		backward_rnn 	= RNN(nb_hidden, return_sequences=True, go_backwards=True, activation=rnn_activation)(maxpooling)
		merge_rnn 		= merge([forward_rnn, backward_rnn], mode='sum', concat_axis=-1)
		drop 			= Dropout(dropout_rate)(merge_rnn)
		outputs 		= TimeDistributed(Dense(output_dim=self.nb_classes, activation='softmax'), name='output_source')(drop)
		self.classifier = Model(input=inputs, output=outputs)

		logger.info('Compiling...')
		self._compile()
		if verbose:
			self._summary()
	
	def _compile(self):
		self.classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy')

	def _summary(self):
		self.classifier.summary()
		logger.debug('Model built: {}'.format(pformat(self.classifier.get_config())))
