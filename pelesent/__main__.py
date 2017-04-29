from __future__ import absolute_import, unicode_literals
import os
import argparse
import logging
import datetime

import numpy as np
import pandas as pd
import gensim
from sklearn.model_selection import KFold
# import nltk

from pelesent import __prog__, __title__, __summary__, __uri__, __version__
from pelesent.log import configure_stream
from pelesent.utils import unroll, vectorize, unvectorize, pad_sequences
from pelesent.models import get_model
from pelesent.embeddings import AvailableEmbeddings
from pelesent.error_analysis import ErrorAnalysis

LOG_DIR 		= 'data/log/'
SUBMISSION_DIR 	= 'data/sub/'
TRAIN_POS_FILE 	= 'data/corpora/buscape2_fixed.pos'
TRAIN_NEG_FILE 	= 'data/corpora/buscape2_fixed.neg'
EMB_TYPE 		= 'word2vec'
EMB_FILE		= 'data/embs/w2v-twitter-skip-300.model'
FOLDS 			= 10
EPOCHS			= 5
BATCH_SIZE 		= 32
MODEL_NAME 		= 'RCNN'
TRAIN_STRATEGY	= 'bucket'
OPT_PARAMS 		= {'nb_hidden':100, 'rnn': 'GRU'}


logger 		= logging.getLogger(__name__)


def load_options():
	parser = argparse.ArgumentParser()
	parser.add_argument('-l', '--load', action='store_true', help='load a trained model')
	parser.add_argument('-s', '--save', action='store_true', help='save a trained model')
	today = 'submition-%s.csv' % datetime.datetime.now().strftime('%y-%m-%d')
	parser.add_argument('-o', '--output', default=today, type=str, help='submition filename')
	parser.add_argument('--gpu', action='store_true', help='run on GPU instead of on CPU')
	return parser.parse_args()


def load_data():
	# detectar risadas?
	data_pos, data_neg = [], []
	vocab = {}
	with open(TRAIN_POS_FILE, 'r', encoding='utf8') as f:
		for line in f:
			data_pos.append(line.strip().split())
			for word in data_pos[-1]:
				if word not in vocab:
					vocab[word] = len(vocab)+1
	with open(TRAIN_NEG_FILE, 'r', encoding='utf8') as f:
		for line in f:
			data_neg.append(line.strip().split())
			for word in data_neg[-1]:
				if word not in vocab:
					vocab[word] = len(vocab)+1
	return data_pos, data_neg, vocab

def load_emb_model():
	emb_model = AvailableEmbeddings.get(EMB_TYPE)()
	emb_model.load(EMB_FILE)
	return emb_model

def data_as_matrix(data_pos, data_neg, vocab, emb_model):
	x = []
	for tks in data_pos:
		x.append(list(map(vocab.__getitem__, tks)))
	for tks in data_neg:
		x.append(list(map(vocab.__getitem__, tks)))
	X = np.array(x)
	Y = vectorize(np.concatenate((np.ones(len(data_pos)), np.zeros(len(data_neg)))))
	return X, Y


def run(options):
	if options.load:
		pass

	logger.info('Loading data...')
	data_pos, data_neg, vocab = load_data()
	nb_tokens = sum(map(len, data_pos)) + sum(map(len, data_neg))
	nb_sents  = len(data_pos) + len(data_neg)
	max_sent_size = max(max(map(len, data_pos)), max(map(len, data_neg)))
	logger.info('Nb tokens: {}'.format(nb_tokens))
	logger.info('Nb sentences: {}'.format(nb_sents))
	logger.info('Pos. sentences: {} ({:.2f})'.format(len(data_pos), len(data_pos)/nb_sents))
	logger.info('Neg. sentences: {} ({:.2f})'.format(len(data_neg), len(data_neg)/nb_sents))
	logger.info('Avg sent size: {:.2f}'.format(nb_tokens / nb_sents))
	logger.info('Max sent size: {}'.format(max_sent_size))
	logger.info('Vocab size: {}'.format(len(vocab)))

	logger.info('Loading embeddings...')
	emb_model = load_emb_model()
	nb_oovs, nb_occur_oovs, top_k_oovs = emb_model.oov_statistics(unroll(data_pos+data_neg), top_k=10)
	logger.info('Nb oovs: {}'.format(nb_oovs))
	logger.info('Nb occur oovs: {}'.format(nb_occur_oovs))
	logger.info('Emb. miss ratio: {:.2f}'.format(nb_oovs / len(vocab)))
	logger.info('Emb. miss occur ratio: {:.2f}'.format(nb_occur_oovs / nb_tokens))
	logger.info('Training vocab size: {}'.format(len(vocab) - nb_oovs + 1))
	logger.info('Transforming data to np arrays...')
	X, Y = data_as_matrix(data_pos, data_neg, vocab, emb_model)

	logger.info('Starting {}-fold cross-validation'.format(FOLDS))
	kf = KFold(n_splits=FOLDS, shuffle=True)

	ea = ErrorAnalysis()
	input_length = max_sent_size if TRAIN_STRATEGY == 'padding' else None
	model = None

	for k, (train_index, test_index) in enumerate(kf.split(X)):
		logger.info('\n---------\n')
		logger.info('K fold: {}'.format(k+1))
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]

		if TRAIN_STRATEGY == 'padding':
			X_train = pad_sequences(X_train, maxlen=max_sent_size)
			X_test = pad_sequences(X_test, maxlen=max_sent_size)

		model = get_model(MODEL_NAME, vocabulary=vocab, emb_model=emb_model, batch_size=BATCH_SIZE,
										input_length=input_length, nb_classes=2, strategy=TRAIN_STRATEGY)
		model.build(**OPT_PARAMS)
		model.train(X_train, Y_train, X_test, Y_test, nb_epoch=EPOCHS, verbose=True)

		Y_pred = model.predict(X_test, verbose=True)
		ea.count(unvectorize(Y_pred), unvectorize(Y_test))
		ea.report()

	if options.save:
		pass


def cli():
	'''Add some useful functionality here or import from a submodule'''

	# load the argument options
	options = load_options()

	if not os.path.exists(SUBMISSION_DIR):
		os.makedirs(SUBMISSION_DIR)

	if not os.path.exists(LOG_DIR):
		os.makedirs(LOG_DIR)

	# configure root logger to print to STDERR
	logger = logging.getLogger(__name__)
	root_logger = configure_stream(level='DEBUG')
	log_formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s]  %(message)s')
	file_handler = logging.FileHandler('{}.log'.format(os.path.join(LOG_DIR, __prog__)))
	file_handler.setFormatter(log_formatter)
	root_logger.addHandler(file_handler)

	# use GPU?
	if options.gpu:
		import theano.sandbox.cuda
		theano.sandbox.cuda.use('gpu')

	run(options)


if __name__ == '__main__':
	cli()