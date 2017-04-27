import unittest
import sys
sys.path.append("..")
from vocab_creator import VocabBuilder


class TestVocabCreator(unittest.TestCase):

	def setUp(self):
		""" Run before every test"""
		self.vb = VocabBuilder()

	def test_constructor(self):
		self.assertIsNotNone(self.vb.wnl)

	def test_read_csv_into_list(self):
		res = self.vb.read_csv_into_list('/cs/home/mn39/Documents/MSciDissertation/resources/vocab_sources/relatedEntities.csv')
		self.assertIsNotNone(res)

	def test_all_sources_to_set(self):
		res = self.vb.all_sources_to_set('/cs/home/mn39/Documents/MSciDissertation/resources/vocab_sources')
		self.assertIsNotNone(res)

	def test_initialise_lexicon(self):
		stems = self.vb._initialise_lexicon({})
		self.assertTrue(not stems)
		stems = self.vb._initialise_lexicon({'nouns':[]})
		self.assertTrue(not stems)
		stems = self.vb._initialise_lexicon({'nouns':['one','two','three']})
		self.assertFalse(not stems)



if __name__ == '__main__':
    unittest.main()