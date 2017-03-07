from nltk.tokenize import TweetTokenizer
import csv
from textblob.classifiers import NaiveBayesClassifier, MaxEntClassifier, DecisionTreeClassifier
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor,FastNPExtractor
import re
import sys
import time
import numpy
from nltk.corpus import stopwords
import string
from nltk import ngrams
import unidecode

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
		self.stopwds = stopwords.words('english') + punctuation + ["via", "...", u'...']
		#self.train()

	def preprocess_tweet(self, text, is_debug=False):
		decoded = self.preprocess_tweet_text(text)
		tokens = self.tokenizer.tokenize(decoded)
		tokens = [tok for tok in tokens if tok not in self.stopwds]
		tokens = [tok for tok in tokens if not tok.startswith("@")]
		tokens = [tok for tok in tokens if not self.url_pattern.match(tok)]
		tokens = [unicode("[MENTION]") if tok.startswith("@") else tok for tok in tokens ]
		tokens = [unicode("[URL]") if self.url_pattern.match(tok) else tok for tok in tokens ]
		tokens = [tok.lower() if not tok.isupper() and not tok.islower() else tok for tok in tokens ]
		return tokens

	def preprocess_tweet_text(self, text):
		try:
			decoded = text.decode("utf-8")
		except AttributeError:
			print text 
			print "You are probably tokenizing your test tweet before giving it to slassifier, don't do that"
			raise Exception
		except UnicodeEncodeError:
			decoded = unidecode.unidecode(text)
			decoded = decoded.decode("utf-8")
		if not isinstance(decoded, unicode):
			print "Something that is not unicode"
			raise Exception
		return decoded

	def ngram_extractor(self, document):
		tokens = self.preprocess_tweet(document)
		return {w:True for w in ngrams(tokens, self.n)}

	def noun_phrase_extractor(self, document):
		""" This is ridiculously slow and should not be used.
		Even with FastNPExtractor ConllExtractor instead of ConllExtractor"""
		blob = TextBlob(document, np_extractor=ConllExtractor())
		return {np: True for np in blob.noun_phrases}

	def get_feature_extractor(self):
		if self.ft_extractor_name == "ngram_extractor":
			return self.ngram_extractor
		elif self.ft_extractor_name == "noun_phrase_extractor":
			return self.noun_phrase_extractor
		else:
			print "Unrecognised feature extractor"
			raise Exception

	def need_to_filter(self, tweet_row, is_debug = False):
		source = tweet_row[2]
		text = tweet_row[3]
		return source != "Sentiment140" or text.startswith("RT")

	def get_train_test_sets(self):
		training_data = []
		testing_data = []
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
						training_data.append((row[3], polarity))
					elif index < self.test_limit:
						testing_data.append(row)
					else:
						return training_data, testing_data
		return training_data, testing_data

	def set_data(self, training_data, testing_data):
		self.training_data = training_data
		self.testing_data = testing_data

	def train(self):
		if not self.training_data:
			s, t = self.get_train_test_sets()
			self.set_data(s, t)
		self.classifier = self.classifier(self.training_data, feature_extractor = self.get_feature_extractor())
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
				predicted = self.classifier.classify(row[3])
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
		print self.classifier.show_informative_features(19)
		for tweet in self.testing_data:
			if not tweet.startswith("RT"):
				predicted = self.classifier.classify(tweet)
				print tweet
				print " - predicted ", predicted
				print "------------------------------------------------"

	def classify_one(self, tweet):
		return self.classifier.classify(tweet)



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

