import csv
import os
import glob
import json
import preprocessor 
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import re
import urllib2
from urlparse import urlparse

class HeadRequest(urllib2.Request):
	def get_method(self): return "HEAD"

def get_real_url(url):
	""" In tweets, all links come in a t.co format. This method returns the actual link. """
	res = urllib2.urlopen(HeadRequest(url))
	return res.geturl()

def write_to_csv(filename, data):
	""" Writes given list of data to a specified file """
	with open(filename, 'w') as output:
		wr = csv.writer(output)
		wr.writerows([data])

def read_in_json(filename):
	""" Reads in json from a file """
	with open(filename) as json_data:
		d = json.load(json_data)
	asciid = {}
	for key in d:
		asciid[key.encode('ascii')] = [item.encode('ascii') for item in d[key]]
	return asciid 

class VocabBuilder():

	def __init__(self):
		self.wnl = WordNetLemmatizer()

	def read_csv_into_list(self, filename):
		""" Read the Google-provided file into list of concepts """
		rows = csv.reader(open(filename), delimiter = ',')
		filtered = [row for row in rows if len(row)==2]
		cleaned = [name for [name,_] in filtered]
		return cleaned

	def all_sources_to_set(self, path):
		""" Reads all Google-provided csvs from a folder and returns a list or unique words. """
		extension = 'csv'
		os.chdir(path)
		vocab = []
		source_files = [i for i in glob.glob('*.{}'.format(extension))]
		for s in source_files:
			vocab.extend(self.read_csv_into_list(s))
		return list(set(vocab))

	def create_local_vocab(self):
		""" This should be done once. Then the resulting vocab should be modified manually."""
		concepts = self.all_sources_to_set('/cs/home/mn39/Documents/MSciDissertation/resources/vocab_sources')
		write_to_csv('/cs/home/mn39/Documents/MSciDissertation/resources/election_vocab.txt', concepts)

	def _initialise_lexicon(self, vocab):
			""" Finds other forms of the same words and their synonyms and adds to the lexicon """
			stems = set()
			for w in vocab['nouns']:
				stem = self.wnl.lemmatize(w)
				stems.add(stem)
				synonyms = self._get_synonyms(w)
				for s in synonyms:
					stem = self.wnl.lemmatize(s)
					stems.add(stem)
			return stems

	def _get_synonyms(self, word):
		""" Returns a list of synonyms of a word or every word in a list. """
		syns = wordnet.synsets(word)
		if isinstance(word, list):
			for w in word:
				syns.extend(wordnet.synsets(w))
		return [l.name().encode('ascii').replace("_", " ").lower() for s in syns for l in s.lemmas()]

	def construct_vocab(self):
		vocab_raw = read_in_json('/cs/home/mn39/Documents/MSciDissertation/resources/election_vocab.txt')
		for a in vocab_raw['abbreviations']:
			print type(a), 'abbreviations'
			break
		vocab_more = self._initialise_lexicon(vocab_raw)
		vocab_raw['stems'] = vocab_more
		self.vocab = vocab_raw
		return self.vocab


def main():
	vb = VocabBuilder()
	vb.construct_vocab()
	print vb.vocab
	
	

if __name__ == '__main__':
	main()