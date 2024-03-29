import numpy as np
import logging
from pprint import pformat

from keras.models import Model
from keras.layers import *
from keras.constraints import maxnorm

from pelesent.models import NeuralNetwork

logger = logging.getLogger(__name__)


class CNN(NeuralNetwork):
    def build(self, nb_filter=100, filter_length=(3, 4, 5), stride=1,
              pool_length=3,
              cnn_activation='relu',
              nb_hidden=200, rnn='LSTM', rnn_activation='sigmoid',
              dropout_rate=0.5, verbose=True):
        logger.info('Building...')
        inputs = []

        sequence = Input(name='input_source', shape=(self.input_length,),
                         dtype='int32')
        embedded = Embedding(self.emb_vocab_size, self.emb_size,
                             input_length=self.input_length,
                             weights=[self.emb_weights])(sequence)
        conv_blocks = []
        for filter in filter_length:
            cnn1d = Convolution1D(nb_filter=nb_filter, filter_length=filter,
                                  activation='relu', subsample_length=stride,
                                  kernel_constraint=maxnorm(3.))(embedded)
            maxp = MaxPool1D(pool_size=self.input_length - filter + 1)(cnn1d)
            maxp = Flatten()(maxp)
            conv_blocks.append(maxp)
        maxp = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        drop = Dropout(dropout_rate)(maxp)
        output = Dense(output_dim=self.nb_classes, activation='softmax',
                       name='output_source')(drop)

        self.classifier = Model(input=[sequence], output=output)
        logger.info('Compiling...')
        self._compile()
        if verbose:
            self._summary()

    def _compile(self):
        self.classifier.compile(optimizer='adadelta',
                                loss='categorical_crossentropy')

    def _summary(self):
        self.classifier.summary()
        logger.debug(
            'Model built: {}'.format(pformat(self.classifier.get_config())))
