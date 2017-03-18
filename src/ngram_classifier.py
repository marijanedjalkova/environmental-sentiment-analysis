from nltk.tokenize import TweetTokenizer
import csv
from textblob.classifiers import NaiveBayesClassifier, MaxEntClassifier, DecisionTreeClassifier
from nltk.classify import SklearnClassifier
from textblob.classifiers import basic_extractor, contains_extractor
from textblob import TextBlob
from textblob.np_extractors import ConllExtractor,FastNPExtractor
from sklearn.svm import SVC, LinearSVC
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
		elif classifier_name == "SVM":
			self.classifier = SVC
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
		self.stopwds = stopwords.words('english') + punctuation + ["via", u'...', '\n', '\t']
		self.weird_unicode_chars = [u'\xc2', u'\xab', u'\xbb', u'..', u'\xe2', u"\u2122"]


	def preprocess_tweet(self, text, is_debug=False):
		if not text:
			return None
		# text should be decoded by this time
		if isinstance(text, unicode):
			tokens = self.tokenizer.tokenize(text)
		elif isinstance(text, list):
			tokens = text
		elif isinstance(text, str):
			print "Text is a string and not unicode - weird"
			raise Exception
		else:
			print "Not sure what this is at all: ", type(text), text
			raise Exception

		tokens = [tok for tok in tokens if tok not in self.stopwds]
		tokens = [tok for tok in tokens if tok not in self.weird_unicode_chars]
		#tokens = [tok for tok in tokens if not tok.startswith("@")]
		#tokens = [tok for tok in tokens if not self.url_pattern.match(tok)]
		tokens = [unicode("[MENTION]") if tok.startswith("@") else tok for tok in tokens ]
		tokens = [unicode("[URL]") if self.url_pattern.match(tok) else tok for tok in tokens ]
		tokens = [tok.lower() if not tok.isupper() and not tok.islower() else tok for tok in tokens ]
		return tokens

	def decode_text(self, text):
		try:
			decoded = text.decode("utf-8")
		except AttributeError:
			print text 
			print "You are probably tokenizing your test tweet before giving it to classifier, don't do that"
			raise Exception
		except UnicodeEncodeError:
			decoded = unidecode.unidecode(text)
			decoded = decoded.decode("utf-8")
		except UnicodeDecodeError:
			return None 
		if not isinstance(decoded, unicode):
			print "Something that is not unicode"
			raise Exception
		return decoded

	def ngram_extractor(self, document):
		# document should be a list of tokens already - i.e. already preprocess_tweet-ed
		if not isinstance(document, list):
			print "This should be a list of tokens already - i.e. already preprocess_tweet-ed"
			raise Exception
		return {w:True for w in ngrams(document, self.n)}

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
						preprocessed = self.preprocess_tweet(self.decode_text(row[3]))
						if preprocessed:
							training_data.append((preprocessed, polarity))
					elif index < self.test_limit:
						testing_data.append(row)
					else:
						return training_data, testing_data
		return training_data, testing_data

	def run_through_data(self):
		""" This outputs the following: 800000, 0, 800000 -> NO NEUTRAL TWEETS """
		with open("/cs/home/mn39/Documents/MSciDissertation/resources/training.1600000.processed.noemoticon.csv") as csvfile:
			data = csv.reader(csvfile)
			index = 0
			p0 = 0
			p2 = 0
			p4 = 0
			for row in data:
				index +=1
				if ((index * 1.0)/1600000 * 100) % 25 == 0:
					print ((index * 1.0)/1600000 * 100), "%"
				polarity = int(row[0])
				if polarity == 0:
					p0 += 1
				elif polarity == 2:
					p2 += 1
				else:
					p4 += 1
			print p0, p2, p4
			print "done"


	def get_train_test_sets2(self):
		training_data = []
		testing_data = []
		with open("/cs/home/mn39/Documents/MSciDissertation/resources/training.1600000.processed.noemoticon.csv") as csvfile:
			#this file has no headers, nothing to skip
			#row[0] is sentiment - 0, 2 or 4
			#row[5] is the tweet
			data = csv.reader(csvfile)
			totalIndex = 0
			trainingPositives = 0
			trainingNegatives = 0
			testingPositives = 0
			testingNegatives = 0
			for row in data:
				totalIndex += 1
				if ((totalIndex * 1.0) / 1600000 * 100) % 25 == 0:
					print ((totalIndex * 1.0) / 1600000 * 100), "%"
				polarity = int(row[0])
				if self.can_add(polarity, trainingPositives, trainingNegatives, self.train_limit):
					preprocessed = self.preprocess_tweet(self.decode_text(row[5]))
					if preprocessed:
						training_data.append((preprocessed, polarity))
						if polarity == 4: trainingPositives += 1
						else: trainingNegatives += 1
				elif self.can_add(polarity, testingPositives, testingNegatives, self.test_limit):
					testing_data.append(row)
					if polarity == 4: testingPositives += 1
					else: testingNegatives += 1
				else:
					if self.data_ready(trainingPositives, trainingNegatives, testingPositives, testingNegatives):
						return training_data, testing_data
					else:
						continue
			print trainingPositives, trainingNegatives
		return training_data, testing_data

	def can_add(self, polarity, positives, negatives, goal):
		""" This is necessary to make training and testing data uniform when it is not sorted automatically. """
		# print "GOAL BY 2", (goal / 2)
		# print " is it? ", (polarity == 0 and negatives >= goal / 2)
		if polarity == 0 and negatives >= goal / 2: 
			return False
		if polarity == 4 and positives >= goal / 2:
			return False
		return True

	def data_ready(self, trP, trN, teP, teN):
		""" Checks if we have reached our goals in both training and testing sets """
		return trP + trN >= self.train_limit and teP + teN >= self.test_limit


	def set_data(self, training_data, testing_data):
		self.training_data = training_data
		self.testing_data = testing_data

	def train(self):
		if not self.training_data:
			s, t = self.get_train_test_sets()
			self.set_data(s, t)
		if self.classifier != SVC:
			self.classifier = self.classifier(self.training_data, feature_extractor = self.get_feature_extractor())
		else:
			self.classifier = SklearnClassifier(LinearSVC()).train(self.to_featureset(self.training_data))
		print "trained"

	def to_featureset(self, training_data):
		return [(self.ngram_extractor(tw), v) for tw, v in training_data]


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
				predicted = self.classifier.classify(self.decode_text(row[3]))
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
				preprocessed = self.preprocess_tweet(self.decode_text(tweet))
				if preprocessed:
					predicted = self.classifier.classify(preprocessed)
					print " - predicted ", predicted
				else:
					print "Could not classify tweet because of decoding problems"
				print tweet
				print "------------------------------------------------"

	def classify_one(self, tweet):
		preprocessed = self.preprocess_tweet(self.decode_text(tweet))
		if preprocessed:
			to_classify = self.ngram_extractor(self.preprocess_tweet(self.decode_text(tweet)))
			return self.classifier.classify(to_classify)
		else:
			print "Could not classify tweet due to decoding problems"
			return -2		


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

