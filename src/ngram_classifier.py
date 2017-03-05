from nltk.tokenize import TweetTokenizer
import csv
from textblob.classifiers import NaiveBayesClassifier, MaxEntClassifier, DecisionTreeClassifier
import re
import sys
import time
import numpy
from nltk.corpus import stopwords
import string
from nltk import ngrams

class Ngram_Classifier:

	def __init__(self, classifier_name, n, train_length, test_length, ft_extractor):
		if classifier_name == "NaiveBayes":
			self.classifier = NaiveBayesClassifier
		elif classifier_name == "MaxEntClassifier":
			self.classifier = MaxEntClassifier
		else:
			self.classifier = DecisionTreeClassifier
		self.n = n
		self.dataset_len = 1578615
		self.train_limit = train_length
		self.test_limit = train_length + test_length
		self.testing_data = []
		self.ft_extractor_name = ft_extractor
		self.tokenizer = TweetTokenizer()
		self.url_pattern = re.compile("(?P<url>https?://[^\s]+)")
		punctuation = list(string.punctuation)
		self.stopwds = stopwords.words('english') + punctuation + ["via"]
		self.train()

	def preprocess_tweet(self, text, is_test=False):
		if isinstance(text, unicode):
			decoded = text
		else:
			decoded = text.decode("utf-8")
		tokens = self.tokenizer.tokenize(decoded)
		tokens = [tok for tok in tokens if tok not in self.stopwds]
		tokens = ["[MENTION]" if tok.startswith("@") else tok for tok in tokens ]
		tokens = ["[URL]" if self.url_pattern.match(tok) else tok for tok in tokens ]
		tokens = [tok.lower() if not tok.isupper() and not tok.islower() else tok for tok in tokens ]
		return tokens

	def ngram_extractor(self, document):
		features = {}
		for w in ngrams(document, self.n):
			features[w]=True
		return features

	def extra(self, document):
		print "extracting"
		return {}

	def get_feature_extractor(self):
		if self.ft_extractor_name == "ngram_extractor":
			return self.ngram_extractor
		else:
			return self.extra

	def need_to_filter(self, tweet_row):
		source = tweet_row[2]
		text = tweet_row[3]
		return source != "Sentiment140" or text.startswith("RT")

	def train(self):
		training_data = []
		with open("/cs/home/mn39/Documents/MSciDissertation/resources/Sentiment-Analysis-Dataset.csv") as csvfile:
			data = csv.reader(csvfile) # 1578615 
			next(data, None) # skip headers
			# format: ItemID, Sentiment, SentimentSource, SentimentText
			for row in data:
				index = int(row[0])
				if ((index * 1.0)/self.train_limit * 100) % 25 == 0:
					print ((index * 1.0)/self.train_limit * 100), "%"
				if not self.need_to_filter(row):
					polarity = int(row[1]) # 0 or 1
					if index < self.train_limit:
						training_data.append((self.preprocess_tweet(row[3]), polarity))
					elif index < self.test_limit:
						self.testing_data.append(row)
					else:
						ft_extractor = self.get_feature_extractor()
						self.classifier = self.classifier(training_data, feature_extractor = ft_extractor)
						break
			print "trained"

	def test(self):
		correct = 0
		kaggle = 0
		to_test = len(self.testing_data)
		for row in self.testing_data:
			index = int(row[0])
			if ( (index * 1.0) / to_test * 100) % 25 == 0:
				print ( (index * 1.0) / to_test * 100), "%"
			if not self.need_to_filter(row):
				polarity = int(row[1])
				tokens = self.preprocess_tweet(row[3])
				predicted = self.classifier.classify(tokens)
				if predicted == polarity:
					correct += 1
				continue
			else:
				kaggle += 1
		accuracy = correct * 1.0/(self.test_limit - self.train_limit - kaggle) 
		print accuracy * 100 , "%"
		return accuracy

	def classify_all(self):
		to_test = len(self.testing_data)
		for tweet in self.testing_data:
			tokens = self.preprocess_tweet(tweet, True)
			predicted = self.classifier.classify(tokens)
			print tweet
			print tokens 
			print " - predicted ", predicted
			print "------------------------------------------------"



def main(argv):
	classifier = argv[0]
	n = int(argv[1])
	learn = int(argv[2])
	test = int(argv[3])
	ft_extractor = argv[4]
	s = time.time()
	nb = Ngram_Classifier(classifier, n, learn, test, ft_extractor)
	nb.test()
	print "Total time: ",  time.time() - s

if __name__=="__main__":
	main(sys.argv[1:])

