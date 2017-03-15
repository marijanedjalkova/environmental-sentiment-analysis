import tweepy
import json
from stream_listener import StreamListener
from ngram_classifier import Ngram_Classifier

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

def main():
	nc1 = Ngram_Classifier("SVM", 1, 300000, 0, "ngram_extractor")
	training, testing = nc1.get_train_test_sets()
	nc1.set_data(training, testing)
	nc1.train()
	#nc1.classifier.show_informative_features(15)
	tt = TwitterTool()
	tweets = tt.search_tweets("grangemouth", 100)
	tweet_list = tt.extract_text_from_tweets(tweets)

	for t in tweet_list:
		if not t.startswith("RT"):
			r1 = nc1.classify_one(t)
			# r2 = nc2.classify_one(t)
			# r3 = nc3.classify_one(t)
			# r4 = nc4.classify_one(t)
			print "Tweet: ", t 
			print r1#, " vs ", r2 #, " vs ", r3, " vs ", r4
			print "========================================================="


if __name__ == '__main__':
	main()
