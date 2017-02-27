from twitter_tool import TwitterTool
import csv
from nltk import word_tokenize

def test_twitter():
	tt = TwitterTool()
	tt.stream(["london"])
	return
	tweets = tt.search_tweets("standrews", 12)
	for tweet in tweets:
		print tweet.created_at, '\n',  tweet.text
		print "--------------"

def get_bag_of_words(string):
	return dict([(word, True) for word in word_tokenize(string)])

def main():
	with open("../resources/Sentiment-Analysis-Dataset.csv") as csvfile:
		data = csv.reader(csvfile) 
		# format ItemID, Sentiment, SentimentSource, SentimentText
		# SentimentSource set is ['Sentiment140', 'Kaggle'] - not sure if it matters

		for row in data:
			text = row[3]
			wordset = get_bag_of_words(text)
			polarity = row[1]	

	


if __name__ == '__main__':
	main()
