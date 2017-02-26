import tweepy
import json

def get_twitter_api():
	with open("auth_twitter.txt") as f:
		d = json.load(f)

	auth = tweepy.OAuthHandler(d["consumer_key"], d["consumer_secret"])
	auth.set_access_token(d["access_token"], d["access_token_secret"])
	return tweepy.API(auth, wait_on_rate_limit=True)

def search(api, query, n):
	tweets = tweet_batch = api.search(q=query, count=n)
	ct = 1
	while len(tweets) < n and ct < 100:
		print "found ", (len(tweets))
		tweet_batch = api.search(q=query, 
								 count=n - len(tweets),
								 max_id=tweet_batch.max_id)
		tweets.extend(tweet_batch)
		ct += 1
	return tweets

def main():
	api = get_twitter_api()

	tweets = search(api, "standrews", 12)
	for tweet in tweets:
		print tweet.created_at, '\n',  tweet.text, '\n\n'
		print "--------------"


if __name__ == '__main__':
	main()
