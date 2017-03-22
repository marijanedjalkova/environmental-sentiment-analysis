import tweepy
import json
from stream_listener import StreamListener
from ngram_classifier import Ngram_Classifier
from senti140 import *
from textblob import TextBlob

class TwitterTool:

	def __init__(self):
		self.api = self.get_twitter_api()

	def get_twitter_api(self):
		with open("/cs/home/mn39/Documents/MSciDissertation/src/auth_twitter.txt") as f:
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

	def stream(self, keywords):
		stream_listener = StreamListener()
		stream_listener.set_twitter_tool(self)
		stream = tweepy.Stream(auth=self.api.auth, listener=stream_listener)
		stream.filter(track=keywords)

	def fits(self, status):
		print "fits() is still to be implemented, returning True for now"
		return True

	def extract_text_from_tweets(self, tweets):
		return [t.text for t in tweets if t.lang=="en"]

	def get_textblob_polarity(self, tweet_text, neg_value, pos_value):
		p = TextBlob(tweet_text).sentiment.polarity
		if p < 0.0:
			return neg_value, p
		elif p > 0.0:
			return pos_value, p
		return (neg_value + pos_value) * 1.0/2, 0.0

def main():
	nc1 = Ngram_Classifier("SVM", 1, 300000, 3000, "ngram_extractor")
	training, testing = nc1.get_train_test_sets2()
	nc1.set_data(training, testing)
	nc1.train()
	nc1.test2()
	return
	#nc1.classifier.show_informative_features(15)
	tt = TwitterTool()
	tweets = tt.search_tweets("grangemouth", 100)
	tweet_list = tt.extract_text_from_tweets(tweets)
	correctCount = 0
	overallCount = 0
	neutralCount = 0
	errors = 0
	notes = []
	for t in tweet_list:
		if not t.startswith("RT"):
			r1 = float(nc1.classify_one(t))
			overallCount += 1
			sentiR, sentiAbsolute = tt.get_textblob_polarity(t, 0.0, 1.0)
			sent140 = get_single_polarity(nc1.preprocess_tweet(nc1.decode_text(t)), 0.0, 4.0)
			if sent140 ==-2:
				errors += 1
				continue
			if r1 == sent140:
				correctCount += 1
			elif abs(r1 - sent140) <= 0.5:
				neutralCount += 1
			else:
				notes += [(r1, sent140, t)]
			print "Tweet: ", t 
			print r1, " vs ", sentiR , " <- ", sentiAbsolute, " vs ", sent140
			print "========================================================="
	print "Overall: ", overallCount
	print "Correct: ", correctCount
	print "Precision: ", correctCount * 100.0/overallCount, "%"
	print "About right: ", neutralCount
	print "Erorrs: ", errors
	for n in notes: print n


if __name__ == '__main__':
	main()
