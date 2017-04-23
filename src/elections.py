from twitter_tool import * 
import json
from vocab_creator import VocabBuilder
from nltk.tokenize import TweetTokenizer


def main():
	d = read_training_data('/cs/home/mn39/Documents/MSciDissertation/resources/election_tweets.txt')
	tm = TopicModel(d)
	tm.train()

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

	def __init__(self, training_data):
		self.vocab = VocabBuilder().construct_vocab()
		self.training_data = training_data

	def train(self):
		for t in self.training_data:
			print self.extract_features(t['text'])

	def extract_features(self, text):
		res = {}
		try:
			parsed = preprocessor.parse(text)
			if parsed.mentions:
				# count how many mentions are known
				mention_list = [o.match for o in parsed.mentions]
				res['mentions'] = len(filter(lambda mention: self.mention_known(mention), mention_list))
			preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.EMOJI, preprocessor.OPT.MENTION)
			cleaned = preprocessor.clean(text)
			tokens = TweetTokenizer().tokenize(cleaned)
			print tokens
			return res	
		except (UnicodeDecodeError, UnicodeEncodeError) as e:
			print text
			print "--------------------!!!"
			return None


	def mention_known(self, mention):
		return mention in self.vocab['mentions']


if __name__ == '__main__':
	main()