from twitter_tool import TwitterTool
from textBlobClassifier import TextBlobClassifier
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
	acc = s.train_and_test()
	print acc
	s = TextBlobClassifier()
	acc2 = s.test()
	print acc, " vs ", acc2


if __name__ == '__main__':
	main()
