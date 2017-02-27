from twitter_tool import TwitterTool
import csv
from nltk import word_tokenize
from nltk import FreqDist

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
	dist = FreqDist()
	with open("../resources/Sentiment-Analysis-Dataset.csv") as csvfile:
		data = csv.reader(csvfile) 
		print sum(1 for row in data)
		return
		# format ItemID, Sentiment, SentimentSource, SentimentText
		# SentimentSource set is ['Sentiment140', 'Kaggle'] - not sure if it matters

		for row in data:
			text = row[3]
			polarity = row[1]
			wordset = get_bag_of_words(text)
			for w in wordset:
				index = polarity if polarity==1 else -1
				dist[w] += index

				

	


if __name__ == '__main__':
	main()
