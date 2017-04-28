import unittest
import sys
sys.path.append("..")
from twitter_tool import *


class TestTwitterTool(unittest.TestCase):

	def setUp(self):
		""" Run before every test"""
		self.tt = TwitterTool()

	def test_constructor(self):
		self.assertIsNotNone(self.tt.api)

	def test_get_textblob_polarity(self):
		r = self.tt.get_textblob_polarity("OMG I am so so happy!", 0, 1)
		self.assertIsNotNone(r)
		self.assertIsInstance(r, tuple)
		self.assertTrue(r[0]>=0)
		self.assertTrue(r[0]<=1)

if __name__ == '__main__':
    unittest.main()