import tweepy
import json
from stream_listener import StreamListener
from ngram_classifier import Ngram_Classifier
from senti140 import *
from textblob import TextBlob

from nltk.tokenize import TweetTokenizer
import re
from nltk.corpus import stopwords
import unidecode
import string
from emojiSentiment import *
from rb_environmental_classifier import *
from dataset import DataSet
from elections import *

class TwitterStreamListener(tweepy.StreamListener):

	def __init__(self, api, twitter_tool):
		self.api = api
		self.twitter_tool = twitter_tool

	def on_status(self, status):
		if self.twitter_tool.is_flagged(status.text):
			print status.text

	def on_error(self, status):
		print status


class TwitterTool:

	def __init__(self):
		self.api = self.get_twitter_api()
		self.sentiment_model = self.get_sentiment_model()
		self.topic_model = self.get_topic_model()	

	def get_sentiment_model(self):
		sentiment_dataset_fname = "../resources/training.1600000.processed.noemoticon.csv"
		sentiment = Ngram_Classifier("SVM", 1, 300000, 0, "preprocessing_extractor")
		sentiment.dataset = DataSet(sentiment_dataset_fname, 0, 5, 4, 0)
		training, testing = sentiment.get_train_test_sets()
		sentiment.set_data(training, testing)
		sentiment.train()
		return sentiment

	def get_topic_model(self):
		topic_data = read_training_data('../resources/election_tweets.txt')
		topic = TopicModel(topic_data, 3)
		topic.set_classifier()
		return topic

	def get_twitter_api(self):
		with open("auth_twitter.txt") as f:
			d = json.load(f)

		auth = tweepy.OAuthHandler(d["consumer_key"], d["consumer_secret"])
		auth.set_access_token(d["access_token"], d["access_token_secret"])
		return tweepy.API(auth, wait_on_rate_limit=True)

	def search_tweets(self, query, n):
		tweets = tweet_batch = self.api.search(q=query, count=n)
		ct = 1
		while len(tweets) < n and ct < 100:
			tweet_batch = self.api.search(q=query, 
									 count=n - len(tweets),
									 max_id=tweet_batch.max_id)
			tweets.extend(tweet_batch)
			ct += 1
		return tweets

	def short_search(self, query, n):
		return self.api.search(q=query, count=n)

	def stream(self, keywords):
		stream_listener = TwitterStreamListener(self.api, self)
		stream = tweepy.Stream(auth=self.api.auth, listener=stream_listener)
		stream.filter(track=keywords)

	def is_flagged(self, data):
		sentiment_featureset = self.sentiment_model.extract_features(data)
		if sentiment_featureset:
			polarity = self.sentiment_model.classify_one(sentiment_featureset)
			if polarity == self.sentiment_model.dataset.negative_value:
				topic_label = self.topic_model.classify(data)[0]
				if topic_label == 1:
					return True
		return False

	def extract_text_from_tweets(self, tweets):
		return [t.text for t in tweets if t.lang=="en"]

	def get_textblob_polarity(self, tweet_text, neg_value, pos_value):
		""" Returns pos_value as the first result if the tweet is neutral. """
		p = TextBlob(tweet_text).sentiment.polarity
		if p < 0.0:
			return neg_value, p
		elif p > 0.0:
			return pos_value, p
		#return (neg_value + pos_value) * 1.0/2, 0.0
		return pos_value, 0.0
		

def main_stream():
	tt = TwitterTool()
	tt.stream(["election"])

if __name__ == '__main__':
	main_stream()
