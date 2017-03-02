from twitter_tool import TwitterTool
from textBlob_classifier import TextBlobClassifier
from very_naive_classifier import SimpleClassifier
from naive_bayes_polarity import NBClassifier
from dataset_analysis import *

def test_twitter():
	tt = TwitterTool()
	tt.stream(["london"])
	return
	tweets = tt.search_tweets("standrews", 12)
	for tweet in tweets:
		print tweet.created_at, '\n',  tweet.text
		print "--------------"

def test_classifiers():
	s = SimpleClassifier(500000, 1000)
	acc = s.train_and_test()
	print acc
	s2 = TextBlobClassifier()
	acc2 = s2.test()
	s3 = NBClassifier(3500, 100)
	acc3 = s3.test()
	print "Simple: ", acc
	print "TextBlob: ", acc2
	print "Naive Bayes: ", acc3

def main():
    print "hello world"

if __name__ == '__main__':
	main()
