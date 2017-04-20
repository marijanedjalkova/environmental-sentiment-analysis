from twitter_tool import * 


def main():
	tt = TwitterTool()
	tweets = tt.search_tweets("election", 1500)
	tweet_list = tt.extract_text_from_tweets(tweets)
	length = len(tweet_list)
	print len(set(tweet_list))==length
	

if __name__ == '__main__':
	main()