from nltk import word_tokenize
import csv
from textblob.classifiers import NaiveBayesClassifier

class NBClassifier:

	def __init__(self):
		self.dataset_len = 1578615
		self.train_limit = 100
		self.test_limit = 110
		self.train()



	def train(self):

		training_data = []
		with open("/cs/home/mn39/Documents/MSciDissertation/resources/Sentiment-Analysis-Dataset.csv") as csvfile:
			data = csv.reader(csvfile) # 1578615 
			next(data, None) # skip headers
			# format ItemID, Sentiment, SentimentSource, SentimentText
			# SentimentSource set is ['Sentiment140', 'Kaggle'] - not sure if it matters
			for row in data:
				
				index = int(row[0])
				if index % 10000 == 0:
					print index
				text = (row[3]).decode("utf-8")
				polarity = int(row[1]) # 0 or 1
				if index < self.train_limit:
					training_data.append((text, polarity))
				else:
					self.cl = NaiveBayesClassifier(training_data)
					break
			print "trained"

	def test(self):

		with open("/cs/home/mn39/Documents/MSciDissertation/resources/Sentiment-Analysis-Dataset.csv") as csvfile:
			data = csv.reader(csvfile) # 1578615 
			next(data, None) # skip headers
			correct = 0
			count = 0
			for row in data:
				
				index = int(row[0])
				if index % 10000 == 0:
					print index
				text = (row[3]).decode("utf-8")
				polarity = int(row[1])
				if self.train_limit < index < self.test_limit:
					count +=1
					predicted = self.cl.classify(text)
					if predicted == polarity:
						correct += 1
					continue
				elif index <= self.train_limit:
					continue
				else:
					accuracy = correct * 1.0/(self.test_limit - self.train_limit) 
					print accuracy * 100 , "%"
					return accuracy
