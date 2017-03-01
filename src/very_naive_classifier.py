from nltk import word_tokenize
import csv
from nltk.tokenize import TweetTokenizer

class SimpleClassifier:

	def __init__(self, train_length, test_length):
		self.dist = {}
                self.tokenizer = TweetTokenizer()
                self.train_limit = train_length
                self.test_limit = train_length + test_length

	def get_bag_of_words(self, string):
		words = self.tokenizer.tokenize(string.decode("utf-8"))
		return dict([(word, True) for word in words])

	def get_polarity(self, text):
		res = 0
		words = self.tokenizer.tokenize(text.decode("utf-8"))
		for w in words:
			res += self.dist[w] if w in self.dist else 0
		if res <= 0:
			return 0 
		return 1

	def train_and_test(self):
		dataset_len = 1578615 
		with open("/cs/home/mn39/Documents/MSciDissertation/resources/Sentiment-Analysis-Dataset.csv") as csvfile:
			data = csv.reader(csvfile) # 1578615 
			next(data, None) # skip headers
			# format ItemID, Sentiment, SentimentSource, SentimentText
			# SentimentSource set is ['Sentiment140', 'Kaggle'] - not sure if it matters
			correct = 0
                        first_time = True
			for row in data:
				
				index = int(row[0])
				if index % 10000 == 0:
					print index
				text = row[3]
				polarity = int(row[1])
				polarity_index = polarity if polarity==1 else -1
				if index < self.train_limit:
					words = self.tokenizer.tokenize(text.decode("utf-8"))
					for w in words:
						if w in self.dist:
							self.dist[w] += polarity_index
						else: 
							self.dist[w] = polarity_index
				elif index < self.test_limit:
                                        if first_time:
                                            first_time=False
                                            self.print_defining_features()
					predicted = self.get_polarity(text)
					if predicted == polarity:
						correct += 1
					continue
				else:
					accuracy = correct * 1.0/(self.test_limit - self.train_limit)
					print correct * 1.0/(self.test_limit - self.train_limit) * 100 , "%"
					return accuracy

        def print_defining_features(self):
            lst = sorted(self.dist.items(), key=lambda x: (-x[1], x[0]))
            print lst[:20]
            print lst[-20:]

