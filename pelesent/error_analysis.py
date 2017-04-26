import logging
import numpy as np

logger = logging.getLogger(__name__)

class ErrorAnalysis:

	def __init__(self):
		self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0

	def count(self, predicted, gold):
		self.tp += np.sum((predicted == 1) & (gold == 1))
		self.tn += np.sum((predicted == 0) & (gold == 0))
		self.fp += np.sum((predicted == 1) & (gold == 0))
		self.fn += np.sum((predicted == 0) & (gold == 1))

	def report(self):
		# https://en.wikipedia.org/wiki/F1_score#Diagnostic_Testing
		logger.debug('----------+--------------+--------------+')
		logger.debug('          |   Pred pos   |   Pred neg   |')
		logger.debug('----------+--------------+--------------+')
		logger.debug('Gold pos  |   %8d   |   %8d   |' % (self.tp, self.fn))
		logger.debug('----------+--------------+--------------+')
		logger.debug('Gold neg  |   %8d   |   %8d   |' % (self.fp, self.tn))
		logger.debug('----------+--------------+--------------+')

		self.nist_er = (self.fn + self.fp)/(self.fn + self.tp) # SER
		self.acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn) 
		logger.debug('SER: %.4f' % self.nist_er)
		logger.debug('Acc: %.4f' % self.acc)
