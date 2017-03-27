from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import ngrams
from textblob import TextBlob
from textblob.classifiers import basic_extractor, contains_extractor
from sklearn.svm import SVC, LinearSVC
import re
import sys
import csv
import time
import string
import unidecode

class Ngram_Classifier:
	ERROR = -2

	def __init__(self, classifier_name, n, train_length, test_length, ft_extractor):
		self.classifier = self.get_classifier(classifier_name)
		self.n = n
		self.train_limit = train_length
		self.test_limit = test_length
		self.ft_extractor_name = ft_extractor	
	

	def get_classifier(self, classifier_name):
		if classifier_name == "NaiveBayes":
			from textblob.classifiers import NaiveBayesClassifier
			return NaiveBayesClassifier
		elif classifier_name == "MaxEntClassifier":
			from textblob.classifiers import MaxEntClassifier
			return MaxEntClassifier
		elif classifier_name == "SVM":
			return SVC
		else:
			from textblob.classifiers import DecisionTreeClassifier
			return DecisionTreeClassifier


	def preprocess_tweet(self, text, is_debug=False):
		""" This tokenizes the tweet and throws away the garbage. """
		tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
		url_pattern = re.compile("(?P<url>https?://[^\s]+)")
		weird_unicode_chars = [u'\xc2', u'\xab', u'\xbb', u'..', u'\xe2', u"\u2122"] + ["via", u'...', '\n', '\t']
		stopwds = stopwords.words('english') + list(string.punctuation) + weird_unicode_chars
		try:
			tokens = tokenizer.tokenize(text)
		except UnicodeDecodeError:
			decoded = self.decode_text(text)
			if not decoded:
				return None
			tokens = tokenizer.tokenize(decoded)

		tokens = [tok for tok in tokens if tok not in stopwds]
		tokens = [tok for tok in tokens if tok not in weird_unicode_chars]
		#tokens = [tok for tok in tokens if not tok.startswith("@")]
		#tokens = [tok for tok in tokens if not url_pattern.match(tok)]
		tokens = [unicode("[MENTION]") if tok.startswith("@") else tok for tok in tokens ]
		tokens = [unicode("[URL]") if url_pattern.match(tok) else tok for tok in tokens ]
		tokens = [tok.lower() if not tok.isupper() and not tok.islower() else tok for tok in tokens ]
		return tokens

	def decode_text(self, text):
		""" Takes in a row text. Returns None only if it cannot possibly work with it. """
		try:
			decoded = text.decode("utf-8")
		except AttributeError:
			print text 
			print "You are probably tokenizing your test tweet before giving it to classifier, don't do that"
			raise Exception
		except UnicodeEncodeError:
			print "UnicodeEncodeError"
			decoded = unidecode.unidecode(text)
			decoded = decoded.decode("utf-8")
		except UnicodeDecodeError:
			print "UnicodeDecodeError, returning None for text"
			print text
			# type of text is always string, and the text is, indeed, incomprehensible
			return None 
		if not isinstance(decoded, unicode):
			print "Something that is not unicode"
			raise Exception
		return decoded

	def ngram_extractor(self, document):
		""" Document takes a raw tweet.
		Returns a feature dict. """
		tokens = self.preprocess_tweet(document)
		if tokens:
			return {w:True for w in ngrams(tokens, self.n)}
		return None

	def noun_phrase_extractor(self, document):
		""" This is ridiculously slow and should not be used.
		Even with FastNPExtractor ConllExtractor instead of ConllExtractor"""
		from textblob.np_extractors import ConllExtractor, FastNPExtractor
		blob = TextBlob(document, np_extractor=ConllExtractor())
		return {np: True for np in blob.noun_phrases}

	def extract_features(self, document):
		extractor = self.get_feature_extractor()
		return extractor(document)

	def get_feature_extractor(self):
		if self.ft_extractor_name == "ngram_extractor":
			return self.ngram_extractor
		elif self.ft_extractor_name == "noun_phrase_extractor":
			return self.noun_phrase_extractor
		else:
			print "Unrecognised feature extractor"
			raise Exception

	def pass_filter(self, tweet_row, is_debug = False):
		""" This was initally designed for the old dataset 
		- Sentiment-Analysis-Dataset.csv, to filter out retweets and Kaggle.
		Can also be used on the new one """
		source = tweet_row[2]
		text = tweet_row[3]
		return source != "Kaggle" and not text.startswith("RT")

	def get_train_test_sets(self, filename, polarity_index, tweet_index, positive_value, negative_value, skip_header=False):
		""" Works with the first dataset - Sentiment-Analysis-Dataset.csv. The data here is already shuffled. """
		training_data = []
		testing_data = []
		file_length = 0
		to_collect = self.train_limit + self.test_limit

		with open(filename) as csvfile:
			data = csv.reader(csvfile) # 1578615 
			if skip_header:
				next(data, None) # skip headers
			# format: ItemID, Sentiment, SentimentSource, SentimentText
			index = 0
			changed = True
			trainingPositives = 0
			trainingNegatives = 0
			testingPositives = 0
			testingNegatives = 0
			for row in data:
				if round(((index * 1.0)/to_collect * 100), 4) % 25 == 0 and changed:
					print round(((index * 1.0)/to_collect * 100), 4), "%"
					changed = False
				if self.pass_filter(row):
					polarity = int(row[polarity_index]) # 0 or 1
					if self.can_add(polarity, trainingPositives, trainingNegatives, self.train_limit, positive_value, negative_value):
						featureset = self.extract_features(row[tweet_index])
						if featureset:
							index += 1
							changed = True
							training_data.append((featureset, polarity))
							if polarity == positive_value: trainingPositives += 1
							else: trainingNegatives += 1
					elif self.can_add(polarity, testingPositives, testingNegatives, (self.test_limit), positive_value, negative_value):
						index += 1
						changed = True
						testing_data.append(row)
						if polarity == positive_value: testingPositives += 1
						else: testingNegatives += 1
						continue
					else:
						if self.data_ready(trainingPositives, trainingNegatives, testingPositives, testingNegatives):
							break
			print trainingPositives, trainingNegatives, " and done with extracting."
		return training_data, testing_data

	def can_add(self, polarity, positives, negatives, goal, positive_value, negative_value):
		""" This is necessary to make training and testing data uniform when it is not sorted automatically. """
		if polarity == negative_value and negatives >= goal / 2: 
			return False
		if polarity == positive_value and positives >= goal / 2:
			return False
		return True

	def data_ready(self, trP, trN, teP, teN):
		""" Checks if we have reached our goals in both training and testing sets """
		return trP + trN >= self.train_limit and teP + teN >= (self.test_limit)


	def set_data(self, training_data, testing_data):
		print len(training_data), "training data is being set"
		self.training_data = training_data
		self.testing_data = testing_data

	def train(self):
		from nltk.classify import SklearnClassifier
		if self.training_data is None:
			s, t = self.get_train_test_sets()
			self.set_data(s, t)
		if self.classifier != SVC and not isinstance(self.classifier, SklearnClassifier):
			self.classifier = self.classifier(self.training_data, feature_extractor = self.get_feature_extractor())
		else:
			self.classifier = SklearnClassifier(LinearSVC()).train(self.training_data)
		print "trained"

	def to_featureset(self, training_data):
		""" Careful, this assumes that feature extractor is ngram extractor """
		ft_ex = self.get_feature_extractor()
		return [(ft_ex(tw), v) for tw, v in training_data if ft_ex(tw)]

	def test(self):
		""" This is for the data in the first dataset """
		correct = 0
		error = 0
		to_test = len(self.testing_data)
		for row in self.testing_data:
			index = int(row[0])
			if ( (index * 1.0) / to_test * 100) % 25 == 0:
				print ( (index * 1.0) / to_test * 100), "%"
			if not self.need_to_filter(row):
				polarity = int(row[1])
				predicted = self.classify_one(row[3], True)
				if predicted == polarity:
					correct += 1
				continue
			else:
				error += 1
		accuracy = correct * 1.0/(self.test_limit - error) 
		print accuracy * 100 , "% "
		return accuracy

	def test_unified(self, polarity_index, tweet_index):
		""" This is for any data """
		correct = 0
		error = 0
		index = 0
		to_test = len(self.testing_data)
		for row in self.testing_data:
			index += 1
			if ( (index * 1.0) / to_test * 100) % 25 == 0:
				print ( (index * 1.0) / to_test * 100), "%"
			polarity = int(row[polarity_index])
			tweet = row[tweet_index]
			predicted = self.classify_one(tweet, True)
			if predicted == Ngram_Classifier.ERROR:
				error += 1
			if predicted == polarity:
				correct += 1
		accuracy = correct * 1.0/(to_test - error) 
		print accuracy * 100 , "% : ", correct, "/", (to_test - error)
		print "Errors: ", error
		return accuracy

	def classify_all(self, toDecode):
		""" Does not validate, just prints out the results """
		for tweet in self.testing_data:
			if not tweet.startswith("RT"):
				predicted = self.classify_one(tweet, toDecode)
				print " - predicted ", predicted
				print tweet
				print "------------------------------------------------"

	def classify_one(self, tweet, toDecode=False):
		""" No validation, just prints the result """
		featureset = self.extract_features(tweet)
		if featureset:
			return self.classifier.classify(featureset)
		return Ngram_Classifier.ERROR		


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

