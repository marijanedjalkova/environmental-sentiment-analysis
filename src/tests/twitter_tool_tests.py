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

if __name__ == '__main__':
    unittest.main()