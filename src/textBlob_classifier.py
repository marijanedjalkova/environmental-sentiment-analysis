import csv
from textblob import TextBlob
import random

class TextBlobClassifier:

	def get_polarity(self, text):
		res = 0
		analysis = TextBlob(text.decode("utf-8"))
		res = 0 if analysis.sentiment.polarity < 0 else 1
		p = analysis.sentiment.polarity
		if p > 0:
			res = 1
		elif p < 0:
			res = 0
		else:
			res = random.randint(0, 1)
		return res

	def test(self):
		test_limit = 10000
		with open("/cs/home/mn39/Documents/MSciDissertation/resources/Sentiment-Analysis-Dataset.csv") as csvfile:
			data = csv.reader(csvfile) # 1578615 
			next(data, None) # skip headers
			# format ItemID, Sentiment, SentimentSource, SentimentText
			# SentimentSource set is ['Sentiment140', 'Kaggle'] - not sure if it matters
			counter = 0
			correct = 0
			error = 0
			for row in data:
				
				index = int(row[0])
				if index % 10000 == 0:
					print index
				t = row[3].strip()
				text = t
				
				polarity = int(row[1])
				polarity_index = polarity if polarity==1 else -1
				if index < test_limit:
					predicted = self.get_polarity(text)
					if predicted == None:
						error += 1
						continue
					if predicted == polarity:
						correct += 1
					continue
				else:
					accuracy = correct * 1.0/(test_limit - error) 
					print accuracy * 100 , "%"
					return accuracy

