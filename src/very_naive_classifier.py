from nltk import word_tokenize
import csv

class SimpleClassifier:

	def __init__(self):
		self.dist = {}

	def get_bag_of_words(self, string):
		words = word_tokenize(string.decode("utf-8"))
		return dict([(word, True) for word in words])

	def get_polarity(self, text):
		res = 0
		words = word_tokenize(text.decode("utf-8"))
		for w in words:
			res += self.dist[w] if w in self.dist else 0
		if res <= 0:
			return 0 
		return 1

	def train_and_test(self):
		dataset_len = 1578615 
		train_limit = dataset_len * 2 /3 / 50 # TODO remove division by 10
		test_limit = train_limit + 1000
		print "train limit", train_limit, ", test limit", test_limit
		with open("/cs/home/mn39/Documents/MSciDissertation/resources/Sentiment-Analysis-Dataset.csv") as csvfile:
			data = csv.reader(csvfile) # 1578615 
			next(data, None) # skip headers
			# format ItemID, Sentiment, SentimentSource, SentimentText
			# SentimentSource set is ['Sentiment140', 'Kaggle'] - not sure if it matters
			correct = 0
			for row in data:
				
				index = int(row[0])
				if index % 10000 == 0:
					print index
				text = row[3]
				polarity = int(row[1])
				polarity_index = polarity if polarity==1 else -1
				if index < train_limit:
					words = word_tokenize(text.decode("utf-8"))
					for w in words:
						if w in self.dist:
							self.dist[w] += polarity_index
						else: 
							self.dist[w] = polarity_index
				elif index < test_limit:
					predicted = self.get_polarity(text)
					if predicted == polarity:
						correct += 1
					continue
				else:
					accuracy = correct * 1.0/(test_limit - train_limit)
					print correct * 1.0/(test_limit - train_limit) * 100 , "%"
					return accuracy

