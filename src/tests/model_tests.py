import unittest
import sys
sys.path.append("..")
from elections import *

class TestElections(unittest.TestCase):

	def setUp(self):
		""" Run before every test"""
		self.tm = TopicModel(read_training_data('/cs/home/mn39/Documents/MSciDissertation/resources/election_tweets.txt'), 2)

	def test_constructor(self):
		self.assertIsNotNone(self.tm.vocab)
		self.assertIsNotNone(self.tm.data)
		self.assertIsNotNone(self.tm.wnl)
		self.assertEquals(self.tm.extractor_index, 2)

	def test_get_data_chunks(self):
		ch = list(self.tm.get_data_chunks(0))
		self.assertEquals(len(ch), 1)
		ch = list(self.tm.get_data_chunks(1))
		self.assertEquals(len(ch), len(self.tm.data))

	def test_set_training_testing_data(self):
		self.tm.set_training_testing_data(0)
		self.assertEquals(len(self.tm.testing_data), len(self.tm.data)*0.1)
		self.assertEquals(len(self.tm.training_data), len(self.tm.data)*0.9)
		self.tm.set_training_testing_data(1)
		self.assertEquals(len(self.tm.testing_data), len(self.tm.data)*0.1)
		self.assertEquals(len(self.tm.training_data), len(self.tm.data)*0.9)
		self.tm.set_training_testing_data(0.5)
		self.assertEquals(len(self.tm.testing_data), len(self.tm.training_data))

	def test_classify(self):
		self.tm.set_classifier()
		c, v = self.tm.classify(unicode("@masha is writing her #dissertation #cslife http://amazing.net"))
		self.assertIsInstance(c, int)
		self.assertIsInstance(v, dict)
		

if __name__ == '__main__':
    unittest.main()