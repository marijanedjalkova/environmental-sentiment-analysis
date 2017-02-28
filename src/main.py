from twitter_tool import TwitterTool

from very_naive_classifier import SimpleClassifier



def test_twitter():
	tt = TwitterTool()
	tt.stream(["london"])
	return
	tweets = tt.search_tweets("standrews", 12)
	for tweet in tweets:
		print tweet.created_at, '\n',  tweet.text
		print "--------------"


def main():
	s = SimpleClassifier()
	s.train_and_test()



if __name__ == '__main__':
	main()
