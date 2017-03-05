import tweepy
import json
from stream_listener import StreamListener
from ngram_classifier import Ngram_Classifier

class TwitterTool:

	def __init__(self):
		self.api = self.get_twitter_api()

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

	def stream(self, keywords):
		stream_listener = StreamListener()
		stream_listener.set_twitter_tool(self)
		stream = tweepy.Stream(auth=self.api.auth, listener=stream_listener)
		stream.filter(track=keywords)

	def fits(self, status):
		print "fits() is still to be implemented, returning True for now"
		return True

	def extract_text_from_tweets(self, tweets):
		return [t.text for t in tweets]

def main():
	nc = Ngram_Classifier("NaiveBayes", 2, 30000, 0, "ngram_extractor")
	tt = TwitterTool()
	tweets = tt.search_tweets("standrews", 55)
	tweet_list = tt.extract_text_from_tweets(tweets)

	nc.testing_data = tweet_list
	nc.classify_all()


if __name__ == '__main__':
	main()
