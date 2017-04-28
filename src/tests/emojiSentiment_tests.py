import unittest
import sys
sys.path.append("..")
from emojiSentiment import *


class TestEmojiSentiment(unittest.TestCase):

	def setUp(self):
		""" Run before every test"""
		self.es = EmojiSentiment()

	def test_constructor(self):
		self.assertIsNotNone(self.es.sentiment)

	def test_get_sentiment(self):
		r = self.es.get_sentiment("bldheskhd")
		self.assertIsNone(r)
		r = self.es.get_sentiment("0x2764")
		self.assertIsNotNone(r)

if __name__ == '__main__':
    unittest.main()