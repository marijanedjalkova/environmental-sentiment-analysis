#!/usr/bin/env python
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk import ngrams
from textblob import TextBlob
from textblob.classifiers import basic_extractor, contains_extractor
from sklearn.svm import SVC, LinearSVC
from emojiSentiment import *
import re
import sys
import csv
import time
import string
import preprocessor 
import random
from KFoldValidator import *
from dataset import DataSet

def empty_extractor(document):
	return document

class Ngram_Classifier:
	ERROR = -2

	def __init__(self, classifier_name, n, train_length, test_length, ft_extractor):
		self.classifier_name = classifier_name 
		self.classifier = self.get_classifier(self.classifier_name)
		self.n = n
		self.train_limit = train_length
		self.test_limit = test_length
		self.ft_extractor_name = ft_extractor
		self.training_data = None 
		self.testing_data = None		

	def get_classifier(self, classifier_name):
		if classifier_name == "NaiveBayes":
			from textblob.classifiers import NaiveBayesClassifier
			return NaiveBayesClassifier
		elif classifier_name == "MaxEntClassifier":
			from textblob.classifiers import MaxEntClassifier
			return MaxEntClassifier
		elif classifier_name == "SVM":
			return SVC
		elif classifier_name == "DecisionTree":
			from textblob.classifiers import DecisionTreeClassifier
			return DecisionTreeClassifier
		else:
			print "Unrecognised classifier"
			raise Exception


	def preprocess_tweet(self, text, is_debug=False):
		""" This tokenizes the tweet and throws away the garbage. """
		tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
		url_pattern = re.compile("(?P<url>https?://[^\s]+)")
		weird_chars = [u'..', "via", u'...', '\n', '\t']
		stopwds = stopwords.words('english') + list(string.punctuation) + weird_chars
		try:
			tokens = tokenizer.tokenize(text)
		except (UnicodeDecodeError, UnicodeEncodeError) as e:
			decoded = self.decode_text(text)
			if not decoded:
				return None
			tokens = tokenizer.tokenize(decoded)
		
		tokens = [tok for tok in tokens if tok not in stopwds]
		tokens = [tok for tok in tokens if tok not in weird_chars]

		#tokens = [tok for tok in tokens if not tok.startswith("@")]
		#tokens = [tok for tok in tokens if not url_pattern.match(tok)]
		tokens = [unicode("$MENTION$") if tok.startswith("@") else tok for tok in tokens ]
		tokens = [unicode("$URL$") if url_pattern.match(tok) else tok for tok in tokens ]
		tokens = [tok.lower() if not tok.isupper() and not tok.islower() else tok for tok in tokens ]
		return tokens

	def decode_text(self, text):
		""" Takes in a row text. Returns None only if it cannot possibly work with it. """
		try:
			decoded = text.decode("utf-8")
		except UnicodeEncodeError:
			# tweets from Twitter search go here
			decoded = text.encode('utf8')
		except UnicodeDecodeError:
			# type of text is always string, and the text is, indeed, incomprehensible
			return None 
		return decoded

	def ngram_extractor(self, document):
		""" Document takes a raw tweet.
		Returns a feature dict. """
		tokens = self.preprocess_tweet(document)
		if tokens:
			return {w:True for w in ngrams(tokens, self.n)}
		return None

	def preprocessing_extractor(self, document):
		""" Takes a raw tweet. Uses preprocessor. """
		res = {}
		# take out weird characters 
		try:
			parsed = preprocessor.parse(document)
		except (UnicodeDecodeError, UnicodeEncodeError) as e:
			# Unicode Encode Error for search results
			document = self.decode_text(document)
			if not document:
				return None 
			try:
				parsed = preprocessor.parse(document)
			except (UnicodeDecodeError, UnicodeEncodeError) as e:
				return None
		if parsed.urls:
			res["$URL$"] = True 
		if parsed.mentions:
			res["$MENTION$"] = True 
		if parsed.emojis:
			for em in parsed.emojis:
				e = em.match.decode("utf-8") 
				stringified = "0x" + repr(e[0])[-6:-1]
				s = EmojiSentiment().get_sentiment(stringified)
				if s:
					if "$EMOJI$" in res:
						res["$EMOJI$"] += s 
					else:
						res["$EMOJI$"] = s
		if parsed.hashtags:
			#this is actually nonexistent
			for h in parsed.hashtags:
				word = h.match[1:]
				if not "$HASHTAG$" in res:
					res["$HASHTAG$"] = []
				res["$HASHTAG$"].append(word)

		preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.EMOJI, preprocessor.OPT.MENTION)
		cleaned = preprocessor.clean(document)
		# send the rest to preprocess 
		tokens = self.preprocess_tweet(cleaned)
		if tokens:
			ns = ngrams(tokens, self.n)
			for n in ns:
				res[n] = True
		if not res:
			return None
		return res


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
		elif self.ft_extractor_name == "preprocessing_extractor":
			return self.preprocessing_extractor
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

	def get_train_test_sets(self):
		""" Works with the first dataset - Sentiment-Analysis-Dataset.csv. The data here is already shuffled. """
		training_data = []
		testing_data = []
		file_length = 0
		to_collect = self.train_limit + self.test_limit

		with open(self.dataset.filename) as csvfile:
			data = csv.reader(csvfile) 
			if self.dataset.skip_header:
				next(data, None) # skip headers

			index = 0
			changed = True
			trainingPositives = 0
			trainingNegatives = 0
			testingPositives = 0
			testingNegatives = 0
			for row in data:
				if changed and round(((index * 1.0)/to_collect * 100), 4) % 25 == 0:
					print round(((index * 1.0)/to_collect * 100), 4), "%"
					changed = False
				if self.pass_filter(row):
					polarity = int(row[self.dataset.polarity_index]) 
					if self.can_add(polarity, trainingPositives, trainingNegatives, self.train_limit):
						featureset = self.extract_features(row[self.dataset.tweet_index])
						if featureset:
							index += 1
							changed = True
							training_data.append((featureset, polarity))
							if polarity == self.dataset.positive_value: trainingPositives += 1
							else: trainingNegatives += 1
					elif self.can_add(polarity, testingPositives, testingNegatives, self.test_limit):
						featureset = self.extract_features(row[self.dataset.tweet_index])
						if featureset:
							index += 1
							changed = True
							testing_data.append((featureset, polarity))
							if polarity == self.dataset.positive_value: testingPositives += 1
							else: testingNegatives += 1
						continue
					else:
						if self.data_ready(trainingPositives, trainingNegatives, testingPositives, testingNegatives):
							break
		return training_data, testing_data

	def get_all_data(self):
		""" Returns all data, shuffled, that can later be split into chunks for k-fold validation """
		result = []
		to_collect = self.train_limit + self.test_limit
		with open(self.dataset.filename) as csvfile:
			data = csv.reader(csvfile) 
			if self.dataset.skip_header:
				next(data, None) # skip headers
			counters = {str(self.dataset.positive_value):0, str(self.dataset.negative_value):0}
			index = 0
			for row in data:
				if ( (index * 100.0) / to_collect) % 25 == 0:
					print ((index * 100.0) / to_collect), "%"
				if self.pass_filter(row) and sum(counters.values())<to_collect:
					polarity = int(row[self.dataset.polarity_index]) 
					if counters[str(polarity)] <= to_collect/2:
						featureset = self.extract_features(row[self.dataset.tweet_index])
						if featureset:
							result.append((featureset, polarity))
							counters[str(polarity)] += 1
							index += 1
				else:
					if sum(counters.values())>=to_collect:
						break
			random.shuffle(result)
			return result


	def can_add(self, polarity, positives, negatives, goal):
		""" This is necessary to make training and testing data uniform when it is not sorted automatically. """
		if polarity == self.dataset.negative_value and negatives >= goal / 2: 
			return False
		if polarity == self.dataset.positive_value and positives >= goal / 2:
			return False
		return True

	def data_ready(self, trP, trN, teP, teN):
		""" Checks if we have reached our goals in both training and testing sets """
		return trP + trN >= self.train_limit and teP + teN >= self.test_limit

	def set_data(self, training_data, testing_data):
		self.training_data = training_data
		self.testing_data = testing_data

	def train(self):
		from nltk.classify import SklearnClassifier
		from textblob.classifiers import NaiveBayesClassifier
		if self.training_data is None:
			s, t = self.get_train_test_sets()
			self.set_data(s, t)
		self.classifier = self.get_classifier(self.classifier_name)
		if self.classifier != SVC and not isinstance(self.classifier, SklearnClassifier):
			print self.classifier
			print "-----------------"
			self.classifier = self.classifier(self.training_data, feature_extractor = empty_extractor)
		else:
			self.classifier = SklearnClassifier(LinearSVC()).train(self.training_data)
		print "Sentiment model trained."

	def to_featureset(self, training_data):
		""" Uses feature extractor to convert a tweet to a dictionary of features. """
		ft_ex = self.get_feature_extractor()
		return [(ft_ex(tw), v) for tw, v in training_data if ft_ex(tw)]

	def test(self):
		""" This is for any dataset. """
		to_test = len(self.testing_data)
		conf = {"tp":0, "fp":0, "tn":0, "fn":0}
		for row in self.testing_data:
			polarity = row[1]
			tweet = row[0]
			predicted = self.classify_one(tweet)
			if predicted == polarity:
				if predicted == self.dataset.positive_value:
					conf["tp"]+=1
				else:
					conf["tn"] += 1
			else:
				if predicted == self.dataset.positive_value:
					conf["fp"]+=1
				else:
					conf["fn"] += 1
		correct = conf["tp"] + conf["tn"]
		accuracy = correct * 100.0/to_test
		recall = conf['tp']*100.0/(conf['tp']+conf['fn'])
		return accuracy, recall

	def classify_one(self, tweet):
		""" No validation, just returns the result. """
		return self.classifier.classify(tweet)		

def test_classifier(classifier, n, learn, test, ft_extractor):
	nb = Ngram_Classifier(classifier, n, learn, test, ft_extractor)
	nb.dataset = DataSet("../resources/training.1600000.processed.noemoticon.csv", 0, 5, 4, 0)
	tr, te = nb.get_train_test_sets()
	nb.set_data(tr, te)
	nb.train()
	print nb.test()

def main(argv):
	classifier = argv[0]
	n = int(argv[1])
	learn = int(argv[2])
	test = int(argv[3])
	ft_extractor = argv[4]
	test_classifier(classifier, n, learn, test, ft_extractor)
	#validate(classifier, n, learn, test, ft_extractor)

def validate_all():
	res = []
	ns = [1, 2, 3]
	learns = [20000, 50000, 75000, 100000, 200000, 300000]
	for n in ns:
		print "N = {}".format(n)
		for learn in learns:
			print "LEARN = {}".format(learn)
			ac, re = validate("NaiveBayes", n, learn, learn/10, "preprocessing_extractor")
			res.append(("NaiveBayes", n, learn, learn/10, "preprocessing_extractor", ac, re))
	with open('sentimentvalidationoutputNB.csv','wb') as out:
		csv_out=csv.writer(out)
		csv_out.writerow(['Classifier','n','training','testing','extractor','accuracy','recall'])
		for row in res:
			csv_out.writerow(row)

	

def validate(classifier, n, learn, test, ft_extractor):
	"""k-fold validation"""
	k = 10
	all_size = learn + test 
	nb = Ngram_Classifier(classifier, n, learn, test, ft_extractor)
	nb.dataset = DataSet("../resources/training.1600000.processed.noemoticon.csv", 0, 5, 4, 0)
	data = nb.get_all_data()
	chunks = list(get_data_chunks(data, all_size, all_size/k))
	accuracies = []
	recalls = []
	for i in range(0,k):
		print "i = {}".format(i)
		tr = []
		te = []
 		[tr.extend(el) for el in chunks[:i]] 
		[tr.extend(el) for el in chunks[(i+1):]] 
		te = chunks[i]
		nb.set_data(tr, te)
		nb.train()
		acc, rec = nb.test()
		accuracies.append(acc)
		recalls.append(rec)
	accuracy = reduce(lambda x, y: x + y, accuracies) / len(accuracies)
	recall = reduce(lambda x, y: x + y, recalls) / len(recalls)
	print accuracy, recall
	return accuracy, recall


if __name__=="__main__":
	#main(sys.argv[1:])
	validate_all()

