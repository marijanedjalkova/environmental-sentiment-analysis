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

	def test_preprocess_tweets(self):
		tweet = unicode("@masha is writing her #dissertation http://awesome.com")
		toks = self.nc.preprocess_tweet(tweet)
		self.assertNotIn("$MENTION$", toks)
		self.assertIn("$URL$", toks)
		self.assertNotIn("is", toks)

	def test_ngram_extractor_empty(self):
		tweet = unicode("")
		extr = self.nc.ngram_extractor(tweet)
		self.assertIsNone(extr)

	def test_ngram_extractor_not_empty(self):
		tweet = unicode("@masha is writing her #dissertation http://awesome.com")
		extr = self.nc.ngram_extractor(tweet)
		self.assertIsNotNone(extr)
		self.assertIn((unicode("writing"),), extr)
		for k in extr:
			self.assertTrue(extr[k])

	def test_preprocessing_extractor_empty(self):
		tweet = unicode("")
		extr = self.nc.ngram_extractor(tweet)
		self.assertIsNone(extr)

	def test_preprocessing_extractor_not_empty(self):
		tweet = unicode("@masha is writing her #dissertation http://awesome.com")
		extr = self.nc.ngram_extractor(tweet)
		self.assertIsNotNone(extr)
		self.assertIn((unicode("writing"),), extr)
		for k in extr:
			self.assertTrue(extr[k])

	def test_get_feature_extractor(self):
		self.nc.ft_extractor_name = "something_unrealistic"
		with self.assertRaises(Exception):
			self.nc.get_feature_extractor()
		self.nc.ft_extractor_name = "preprocessing_extractor"

	def test_pass_filter(self):
		self.assertFalse(self.nc.pass_filter([None,None,'Kaggle',"RT blah"]))
		self.assertFalse(self.nc.pass_filter([None,None,'Kaggle',"Not RT blah"]))
		self.assertFalse(self.nc.pass_filter([None,None,'NotKaggle',"RT blah"]))
		self.assertTrue(self.nc.pass_filter([None,None,'NotKaggle',"Not RT blah"]))

	def test_can_add(self):
		self.assertFalse(self.nc.can_add(0,100,100,201,1,0))
		self.assertTrue(self.nc.can_add(0,100,100,202,1,0))
		self.assertFalse(self.nc.can_add(1,100,100,201,1,0))
		self.assertTrue(self.nc.can_add(1,100,100,202,1,0))
		self.assertTrue(self.nc.can_add(0.5,100,100,200,1,0))

	def test_data_ready(self):
		self.assertFalse(self.nc.data_ready(80,200,10,10))
		self.assertTrue(self.nc.data_ready(800,200,10,10))
		self.assertFalse(self.nc.data_ready(80,200,1,10))
		self.assertTrue(self.nc.data_ready(800,200,10,10))

	def test_set_data(self):
		self.assertFalse(hasattr(self.nc,"training_data"))
		self.assertFalse(hasattr(self.nc,"testing_data"))
		self.nc.set_data([1,2,3],[4,5])
		self.assertListEqual(self.nc.training_data, [1,2,3])
		self.assertListEqual(self.nc.testing_data, [4,5])



if __name__ == '__main__':
    unittest.main()