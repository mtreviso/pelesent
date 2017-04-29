from abc import ABCMeta, abstractmethod
import sys
import numpy as np
import logging

np.random.seed(1)

from sklearn.utils import compute_class_weight
from pelesent.utils import unvectorize, unpad_sequences, bucketize, reorder_buckets
from keras.callbacks import EarlyStopping

logger = logging.getLogger(__name__)

class NeuralNetwork(metaclass=ABCMeta):

	def __init__(self, vocabulary=None, emb_model=None, input_length=None, nb_classes=2, strategy='bucket', batch_size=32):
		'''
		:param vocabulary: a dict where keys are words
		:param input_length: sentence size, a value if fixed (padding), or None if variable
		:param nb_classes: number of targets, usually max(y) + 1
		'''
		self.vocabulary = vocabulary
		self.emb_model = emb_model
		self.input_length = input_length
		self.nb_classes = nb_classes
		self.strategy = strategy
		self.batch_size = batch_size
		self.classifier = None
		self._prepare_params()


	def _get_batch(self, data_in, data_out, lengths, data_by_length, shuffle=True, kind='train'):
		while True: # a new epoch
			if shuffle:
				np.random.shuffle(lengths)
			for length in lengths:
				indexes = data_by_length[length]
				if shuffle:
					np.random.shuffle(indexes)
				input_data 	= {'input_source': np.array([data_in[i] for i in indexes])}
				output_data = {'output_source': np.array([data_out[i] for i in indexes])}
				yield (input_data, output_data)
			if kind == 'predict':
				break

	def _prepare_params(self):
		vocab_size = max(self.vocabulary.values()) + 1
		emb_dim = self.emb_model.dimensions

		limit = np.sqrt(6 / (vocab_size + emb_dim)) # glorot_uniform
		self.emb_weights = np.random.uniform(-limit, limit, size=(vocab_size, emb_dim))
		for word, index in self.vocabulary.items():
			self.emb_weights[index] = self.emb_model.get_vector(word)
		self.emb_vocab_size = self.emb_weights.shape[0]
		self.emb_size = self.emb_weights.shape[1]

	def save_weights(self, filename):
		self.classifier.save_weights(filename)

	def load_weights(self, filename):
		self.classifier.load_weights(filename)

	@abstractmethod
	def build(self, **kwargs):
		pass

	def train(self, X_train, Y_train, X_val, Y_val, nb_epoch=20, verbose=True):
		logger.info('Calculating class weight...')
		classes = list(range(self.nb_classes))
		outputs = unvectorize(Y_train)
		class_weight = dict(zip(classes, compute_class_weight('balanced', classes, outputs)))

		callbacks = []
		callbacks.append(EarlyStopping(monitor='val_loss', patience=3, mode='auto'))

		logger.info('Fitting...')
		if self.strategy == 'bucket':
			print('bucketizing train data')
			train_buckets 	= bucketize(X_train, max_bucket_size=self.batch_size)
			train_generator = self._get_batch(X_train, Y_train, *train_buckets, shuffle=True)
			print('bucketizing val data')
			val_buckets 	= bucketize(X_val, max_bucket_size=self.batch_size)
			val_generator	= self._get_batch(X_val, Y_val, *val_buckets, shuffle=False)
			print('fitting :)')
			self.classifier.fit_generator(train_generator, steps_per_epoch=len(train_buckets[0]),
												validation_data=val_generator, validation_steps=len(val_buckets[0]),
												epochs=nb_epoch, class_weight=class_weight, callbacks=callbacks)
		else:
			self.classifier.fit(X_train, Y_train, epochs=nb_epoch, batch_size=self.batch_size, 
									validation_data=(X_val, Y_val), class_weight=class_weight, callbacks=callbacks)


	def _predict_on_batch(self, generator, val_samples, verbose=False):
		# !!!fix keras predict_generator function and make a pull request!!!
		# preds = self.classifier.predict_generator(generator, val_samples=val_samples)
		preds = []
		if verbose:
			print('')
		for i, (X, Y) in enumerate(generator):
			if verbose:
				sys.stdout.write('Prediction %d/%d \r' % (i+1, val_samples))
				sys.stdout.flush()
			out = self.classifier.predict_on_batch(X)
			for y in out:
				preds.append(y)
		return preds

	def predict(self, X_test, verbose=False):
		if self.strategy == 'bucket':
			lengths, data_by_length = bucketize(X_test, max_bucket_size=self.batch_size)
			pred_generator = self._get_batch(X_test, X_test, lengths, data_by_length, shuffle=False, kind='predict')
			preds = self._predict_on_batch(pred_generator, len(data_by_length), verbose=verbose)
			preds, _ = reorder_buckets(preds, preds, lengths, data_by_length)
		else:
			preds = self.classifier.predict(X_test, batch_size=self.batch_size, verbose=verbose)
			preds = unpad_sequences(preds)
		return np.array(preds)
