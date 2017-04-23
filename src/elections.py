from twitter_tool import * 
import json
from vocab_creator import VocabBuilder
from nltk.tokenize import TweetTokenizer
import re
from sklearn.svm import SVC, LinearSVC
from nltk.classify import SklearnClassifier


def main():
	d = read_training_data('/cs/home/mn39/Documents/MSciDissertation/resources/election_tweets.txt')
	tm = TopicModel(d)
	tm.set_classifier()
	tm.test()

def read_training_data(filename):
	""" Reads in training data. """
	with open(filename) as json_data:
		d = json.load(json_data)
	return d 

def get_save_tweets(filename):
	""" Gets tweets from Twitter and saves them as dicts to be labelled. """
	tt = TwitterTool()
	tweets = tt.search_tweets("election", 1500)
	tweet_list = tt.extract_text_from_tweets(tweets)
	data = []
	for tw in set(tweet_list):
		dct = {}
		dct["label"] = 0
		dct["text"] = tw
		data.append(dct)
	with open(filename, 'w') as outfile:
		json.dump(data, outfile)

class TopicModel:

	def __init__(self, data):
		self.vocab = VocabBuilder().construct_vocab()
		self.data = data #882 tweets
		self.errors = 0
		self.set_training_testing_data(0.9)

	def test(self):
		count = 0
		correct = 0
		for dct in self.testing_data:
			count += 1
			res = self.classify(dct['text'])
			if res == dct['label']:
				correct += 1
		print "{}/{}={}%".format(correct, count, (correct*100.0/count))

	def set_training_testing_data(self, portion):
		border_index = int(round(len(self.data)*portion))
		self.testing_data = self.data[:border_index]
		self.training_data = self.data[border_index:]

	def set_classifier(self):
		formatted_data = self.get_feature_vectors()
		self.classifier = SklearnClassifier(LinearSVC()).train(formatted_data)

	def get_feature_vectors(self):
		vector_lst = []
		for dct in self.training_data:
			features = self.extract_features(dct['text'])
			vector_lst.append((features, dct['label']))
		return vector_lst

	def classify(self, text):
		vector = self.extract_features(text)
		return self.classifier.classify(vector)


	def extract_features(self, text):
		res = {}
		try:
			# it's not sentiment analysis so we just need text
			text_str = text.encode('ascii', 'ignore')
			cleaned, mention_dict = self.process_parsing(text_str)
			res.update(mention_dict)
			tokens = TweetTokenizer().tokenize(cleaned)			
			res.update(self.process_tokens(tokens))
			return res	
		except (UnicodeDecodeError, UnicodeEncodeError) as e:
			print text
			print "--------------------!!!"
			self.errors += 1
			return None

	def process_tokens(self, tokens):
		res = {}
		for t in tokens:
			for key in self.vocab.keys():
				if not key in res:
					res[key] = 0
				if self.check_vocab(t, self.vocab[key]):
					res[key]+=1
		return res

	def check_vocab(self, token, wordlist):
		""" Token is one word but word can be a concept consisting of 2 wds or 
		a concept with an underscore"""
		for word in map(str.lower, wordlist):
			if " " in word:
				wds = re.split(' ',word)
				if token in map(str.lower, wds):
					return True 
			else:
				if token == word.lower:
					return True 
		return False

	def process_parsing(self, text_str):
		res = {'mentions':0}
		parsed = preprocessor.parse(text_str)
		if parsed.mentions:
			# count how many mentions are known
			mention_list = [o.match for o in parsed.mentions]
			res['mentions'] = len(filter(lambda mention: self.mention_known(mention), mention_list))
		preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.EMOJI, preprocessor.OPT.MENTION)
		cleaned = preprocessor.clean(text_str)
		return cleaned, res

	def mention_known(self, mention):
		return mention in self.vocab['mentions']

if __name__ == '__main__':
	main()