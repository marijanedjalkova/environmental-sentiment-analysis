import tweepy
import json

def main():
	with open("auth_twitter.txt") as f:
		d = json.load(f)

	consumer_key = d['consumer_key']

	auth = tweepy.OAuthHandler(d["consumer_key"], d["consumer_secret"])
	auth.set_access_token(d["access_token"], d["access_token_secret"])
	api = tweepy.API(auth)
	results = api.search(q="Mice")
	for result in results:
		print result.text



if __name__ == '__main__':
	main()
