import unittest
import sys
sys.path.append("..")
from ngram_classifier import *


class TestNgramClassifier(unittest.TestCase):

	def setUp(self):
		""" Run before every test"""
		self.nc = Ngram_Classifier("SVM", 1, 1000, 20, "preprocessing_extractor")


	def test_constructor(self):
		self.assertIsNotNone(self.nc.classifier)
		self.assertIsNotNone(self.nc.n)
		self.assertIsNotNone(self.nc.train_limit)
		self.assertIsNotNone(self.nc.test_limit)
		self.assertIsNotNone(self.nc.ft_extractor_name)

	def test_get_classifier_NB(self):
		from textblob.classifiers import NaiveBayesClassifier
		c = self.nc.get_classifier("NaiveBayes")
		self.assertIsNotNone(c)
		self.assertIsInstance(NaiveBayesClassifier, type(c))

	def test_get_classifier_MEC(self):
		from textblob.classifiers import MaxEntClassifier
		c = self.nc.get_classifier("MaxEntClassifier")
		self.assertIsNotNone(c)
		self.assertIsInstance(MaxEntClassifier, type(c))

	def test_get_classifier_DT(self):
		from textblob.classifiers import DecisionTreeClassifier
		c = self.nc.get_classifier("DecisionTree")
		self.assertIsNotNone(c)
		self.assertIsInstance(DecisionTreeClassifier, type(c))

	def test_get_classifier_SVM(self):
		from sklearn.svm import SVC, LinearSVC
		c = self.nc.get_classifier("SVM")
		self.assertIsNotNone(c)
		self.assertIsInstance(SVC, type(c))

	def test_get_classifier_UNK(self):
		with self.assertRaises(Exception):
			self.nc.get_classifier("someotherclassifier")

if __name__ == '__main__':
    unittest.main()