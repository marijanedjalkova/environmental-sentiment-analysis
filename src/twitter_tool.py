import tweepy
import json

class TwitterTool:

	def __init__(self):
		print "Creating twitter tool"
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
			print "found ", (len(tweets))
			tweet_batch = self.api.search(q=query, 
									 count=n - len(tweets),
									 max_id=tweet_batch.max_id)
			tweets.extend(tweet_batch)
			ct += 1
		return tweets

