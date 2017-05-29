from __future__ import absolute_import, unicode_literals
import os
import argparse
import logging
import datetime

import numpy as np
np.random.seed(1)

from sklearn.model_selection import StratifiedKFold, train_test_split

from pelesent import __prog__, __title__, __summary__, __uri__, __version__
from pelesent.log import configure_stream
from pelesent.utils import unroll, vectorize, unvectorize, pad_sequences
from pelesent.models import get_model
from pelesent.embeddings import AvailableEmbeddings
from pelesent.error_analysis import ErrorAnalysis


LOG_DIR 		= 'data/log/'
SUBMISSION_DIR 	= 'data/sub/'
FOLDS 			= 10
EPOCHS			= 10
BATCH_SIZE 		= 128
MODEL_NAME 		= 'RCNN'
TRAIN_STRATEGY	= 'bucket'
OPT_PARAMS 		= {'nb_hidden':100, 'rnn': 'GRU'}
MIN_SENT_SIZE 	= 4 

logger 		= logging.getLogger(__name__)


def load_options():
	parser = argparse.ArgumentParser()
	parser.add_argument('--pos-file', type=str, help='pos file', required=True)
	parser.add_argument('--neg-file', type=str, help='neg file', required=True)
	parser.add_argument('--emb-file', type=str, help='emb file', required=True)
	parser.add_argument('--emb-type', type=str, help='emb type', required=True)
	parser.add_argument('--test-pos-file', type=str, help='test pos file', default=None)
	parser.add_argument('--test-neg-file', type=str, help='test neg file', default=None)
	parser.add_argument('-l', '--load', action='store_true', help='load a trained model')
	parser.add_argument('-s', '--save', action='store_true', help='save a trained model')
	parser.add_argument('-b', '--batch-size', type=int, default=128, help='batch size')
	today = 'submition-%s.csv' % datetime.datetime.now().strftime('%y-%m-%d')
	parser.add_argument('-o', '--output', default=today, type=str, help='submition filename')
	parser.add_argument('--gpu', action='store_true', help='run on GPU instead of on CPU')
	return parser.parse_args()


def load_data(pos_file, neg_file, vocab, mss=4):
	# detectar risadas?
	data_pos, data_neg = [], []
	with open(pos_file, 'r', encoding='utf8') as f:
		for line in f:
			tks = line.strip().split()
			if len(tks) >= mss:
				# if tks[0] == 'rt':
				# 	continue
				data_pos.append(tks)
				for word in data_pos[-1]:
					if word not in vocab:
						vocab[word] = len(vocab)+1
	with open(neg_file, 'r', encoding='utf8') as f:
		for line in f:
			tks = line.strip().split()
			if len(tks) >= mss:
				# if tks[0] == 'rt':
				# 	continue
				data_neg.append(tks)
				for word in data_neg[-1]:
					if word not in vocab:
						vocab[word] = len(vocab)+1
	return data_pos, data_neg

def load_emb_model(options):
	emb_model = AvailableEmbeddings.get(options.emb_type)()
	emb_model.load(options.emb_file)
	return emb_model

def data_as_matrix(data_pos, data_neg, vocab, emb_model):
	x = []
	for tks in data_pos:
		x.append(list(map(vocab.__getitem__, tks)))
	for tks in data_neg:
		x.append(list(map(vocab.__getitem__, tks)))
	X = np.array(x)
	pos = np.ones(len(data_pos), dtype=np.int)
	neg = np.zeros(len(data_neg), dtype=np.int)
	Y = vectorize(np.concatenate((pos, neg)))
	return X, Y


def show_stats(data_pos, data_neg, emb_model, vocab):
	nb_tokens = sum(map(len, data_pos)) + sum(map(len, data_neg))
	nb_sents  = len(data_pos) + len(data_neg)
	min_sent_size = min(min(map(len, data_pos)), min(map(len, data_neg)))
	max_sent_size = max(max(map(len, data_pos)), max(map(len, data_neg)))
	logger.info('Nb tokens: {}'.format(nb_tokens))
	logger.info('Nb sentences: {}'.format(nb_sents))
	logger.info('Pos. sentences: {} ({:.2f})'.format(len(data_pos), len(data_pos)/nb_sents))
	logger.info('Neg. sentences: {} ({:.2f})'.format(len(data_neg), len(data_neg)/nb_sents))
	logger.info('Avg sent size: {:.2f}'.format(nb_tokens / nb_sents))
	logger.info('Min sent size: {}'.format(min_sent_size))
	logger.info('Max sent size: {}'.format(max_sent_size))
	logger.info('Vocab size: {}'.format(len(vocab)))

	data_all = unroll(data_pos+data_neg)
	nb_oovs, nb_occur_oovs, top_k_oovs = emb_model.oov_statistics(data_all, top_k=10)
	logger.info('Embedding stats:')
	logger.info('Nb oovs: {}'.format(nb_oovs))
	logger.info('Nb occur oovs: {}'.format(nb_occur_oovs))
	logger.info('Emb. miss ratio: {:.2f}'.format(nb_oovs / len(vocab)))
	logger.info('Emb. miss occur ratio: {:.2f}'.format(nb_occur_oovs / nb_tokens))
	logger.info('Vocab size: {}'.format(len(vocab) - nb_oovs + 1))


def run(options):
	if options.load:
		pass

	vocab = {}

	logger.info('Loading data...')
	data_pos, data_neg = load_data(options.pos_file, options.neg_file, vocab, mss=MIN_SENT_SIZE)

	logger.info('Loading embeddings...')
	emb_model = load_emb_model(options)

	logger.info('Train stats:')
	show_stats(data_pos, data_neg, emb_model, vocab)

	if options.test_pos_file is not None:
		test_data_pos, test_data_neg = load_data(options.test_pos_file, options.test_neg_file, vocab, mss=1)

		logger.info('Test stats:')
		show_stats(test_data_pos, test_data_neg, emb_model, vocab)

		logger.info('Transforming data to np arrays...')
		X, Y = data_as_matrix(data_pos, data_neg, vocab, emb_model)
		X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.1, random_state=1, stratify=unvectorize(Y))
		X_test, Y_test = data_as_matrix(test_data_pos, test_data_neg, vocab, emb_model)

		input_length = max_sent_size if TRAIN_STRATEGY == 'padding' else None
		model = get_model(MODEL_NAME, vocabulary=vocab, emb_model=emb_model, batch_size=BATCH_SIZE, input_length=input_length, nb_classes=2, strategy=TRAIN_STRATEGY)
		model.build(**OPT_PARAMS)
		model.train(X_train, Y_train, X_val, Y_val, nb_epoch=EPOCHS, verbose=True)

		Y_pred = model.predict(X_test, verbose=True)
		
		ea = ErrorAnalysis()
		ea.count(unvectorize(Y_pred), unvectorize(Y_test))
		ea.report()


	else:
		logger.info('Transforming data to np arrays...')
		X, Y = data_as_matrix(data_pos, data_neg, vocab, emb_model)
		ea = ErrorAnalysis()
		input_length = max_sent_size if TRAIN_STRATEGY == 'padding' else None
		model = None
		
		logger.info('Starting {}-fold cross-validation'.format(FOLDS))
		kf = StratifiedKFold(n_splits=FOLDS, shuffle=True)
		for k, (train_index, test_index) in enumerate(kf.split(X, unvectorize(Y))):
			logger.info('\n---------\n')
			logger.info('K fold: {}'.format(k+1))
			X_train, X_test = X[train_index], X[test_index]
			Y_train, Y_test = Y[train_index], Y[test_index]

			if TRAIN_STRATEGY == 'padding':
				X_train = pad_sequences(X_train, maxlen=max_sent_size)
				X_test = pad_sequences(X_test, maxlen=max_sent_size)

			model = get_model(MODEL_NAME, vocabulary=vocab, emb_model=emb_model, batch_size=BATCH_SIZE, input_length=input_length, nb_classes=2, strategy=TRAIN_STRATEGY)
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

	BATCH_SIZE = int(options.batch_size)

	# log_filename = __prog__
	f = lambda x: x.split('/')[-1].split('.')[0]
	log_filename = f(options.pos_file) + '-' + f(options.neg_file) + '-' + f(options.emb_file)

	if options.test_pos_file is not None:
		log_filename = f(options.pos_file) + '-' + f(options.test_pos_file) + '-' + f(options.emb_file)

	# configure root logger to print to STDERR
	logger = logging.getLogger(__name__)
	root_logger = configure_stream(level='DEBUG')
	log_formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s]  %(message)s')
	file_handler = logging.FileHandler('{}.log'.format(os.path.join(LOG_DIR, log_filename)))
	file_handler.setFormatter(log_formatter)
	root_logger.addHandler(file_handler)

	# use GPU?
	if options.gpu:
		import theano.sandbox.cuda
		theano.sandbox.cuda.use('gpu')

	run(options)


if __name__ == '__main__':
	cli()