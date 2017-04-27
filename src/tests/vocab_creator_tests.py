import unittest
import sys
sys.path.append("..")
from vocab_creator import VocabBuilder


class TestVocabCreator(unittest.TestCase):

	def setUp(self):
		""" Run before every test"""
		print "setup"
		self.vb = VocabBuilder()

	def test_constructor(self):
		self.assertNotEqual(self.vb.wnl, None)

	def test_lower(self):
		self.assertEqual('foo'.upper(), 'FOO')




if __name__ == '__main__':
    unittest.main()