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
		self.assertEquals(len(ch),1)
		ch = list(self.tm.get_data_chunks(1))
		self.assertEquals(len(ch),len(self.tm.data))



if __name__ == '__main__':
    unittest.main()