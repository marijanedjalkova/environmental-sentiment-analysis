from twitter_tool import TwitterTool
import csv
from nltk import word_tokenize

dist = {}

def test_twitter():
	tt = TwitterTool()
	tt.stream(["london"])
	return
	tweets = tt.search_tweets("standrews", 12)
	for tweet in tweets:
		print tweet.created_at, '\n',  tweet.text
		print "--------------"

def get_bag_of_words(string):
	words = []
	try:
		words = word_tokenize(string)
	except UnicodeDecodeError:
		pass
	return dict([(word, True) for word in words])

def get_polarity(text):
	res = 0
	try:
		words = word_tokenize(text)
		for w in words:
			res += dist[w] if w in dist else 0
	except UnicodeDecodeError:
		pass
	if res < 0:
		return 0 
	return 1

def main():
	
	dataset_len = 1578615 
	test_limit = dataset_len * 2 /3 / 10 # TODO remove division by 10
	with open("../resources/Sentiment-Analysis-Dataset.csv") as csvfile:
		data = csv.reader(csvfile) # 1578615 
		next(data, None) # skip headers
		# format ItemID, Sentiment, SentimentSource, SentimentText
		# SentimentSource set is ['Sentiment140', 'Kaggle'] - not sure if it matters
		counter = 0
		total = 0
		for row in data:
			
			index = int(row[0])
			if index % 10000 == 0:
				print index
			text = row[3]
			polarity = int(row[1])
			polarity_index = polarity if polarity==1 else -1
			if index < test_limit:
				try:
					words = word_tokenize(text)
					for w in words:
						if w in dist:
							dist[w] += polarity_index
						else: dist[w] = polarity_index
				except UnicodeDecodeError:
					pass
			else:
				
				counter += 1
				if counter < 1000:
					predicted = get_polarity(text)
					if predicted == polarity:
						total += 1
					continue
				print total/10 , "%"
				return

				

	


if __name__ == '__main__':
	main()
