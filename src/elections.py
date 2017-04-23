from twitter_tool import * 
import json
from vocab_creator import VocabBuilder


def main():
	d = read_training_data('/cs/home/mn39/Documents/MSciDissertation/resources/election_tweets.txt')
	tm = TopicModel()

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

	def __init__(self):
		self.vocab = VocabBuilder().construct_vocab()


	

if __name__ == '__main__':
	main()