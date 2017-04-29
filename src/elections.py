from twitter_tool import * 
import json
from vocab_creator import VocabBuilder
from nltk.tokenize import TweetTokenizer
import re
from sklearn.svm import SVC, LinearSVC
from nltk.classify import SklearnClassifier
from nltk.corpus import stopwords


def main():
	d = read_training_data('/cs/home/mn39/Documents/MSciDissertation/resources/election_tweets.txt')
	for n in range(1,4):
		print "n={}".format(n)
		tm = TopicModel(d, n)
		#tm.set_classifier()
		#tm.test()
		tm.kfold_validation(10)

def read_training_data(filename):
	""" Reads in training data. """
	with open(filename) as json_data:
		d = json.load(json_data)
	return d 

def get_save_tweets(filename):
	""" Gets tweets from Twitter and saves them as dicts to be labelled. """
	tt = TwitterTool()
	tweets = tt.search_tweets("election", 1500)
	tweet_list = tt.extract_text_from_tweets(tweets)
	data = []
	for tw in set(tweet_list):
		dct = {}
		dct["label"] = 0
		dct["text"] = tw
		data.append(dct)
	with open(filename, 'w') as outfile:
		json.dump(data, outfile)

class TopicModel:

	def __init__(self, data, extractor_index):
		self.extractor_index = extractor_index
		self.vocab = VocabBuilder().construct_vocab()
		self.data = data #882 tweets
		self.errors = 0
		self.wnl = WordNetLemmatizer()
		self.set_training_testing_data(0.9)

	def analyse_dataset(self, dataset):
		""" This just shows how many pos/neg tweets there are in the set """
		c = {'p':0, 'n':0}
		for i in dataset:
			if i['label'] == 1:
				c['p']+=1
			else:
				c['n']+=1
		print c

	def get_data_chunks(self, n):
		""" Produces equaly sized chunks of size n. Throws away the remainder. """
		if n == 0:
			n = len(self.data)
		for i in range(0, len(self.data), n):
			if len(self.data[i:i + n]) == n:
				yield self.data[i:i + n]

	def kfold_validation(self, k):
		""" Prints accuracy and recall at the end """
		size = len(self.data) / k 
		if size <= 0:
			k = 1
			size = len(self.data)
		chunks = list(self.get_data_chunks(size))
		accuracies = []
		recalls = []
		for i in range(k):
			print "i = {}".format(i)
			self.training_data = []
			[self.training_data.extend(el) for el in chunks[:i]] 
			[self.training_data.extend(el) for el in chunks[(i+1):]] 
			self.testing_data = chunks[i]
			self.set_classifier()
			acc, rec = self.test()
			accuracies.append(acc)
			recalls.append(rec)
		print reduce(lambda x, y: x + y, accuracies) / len(accuracies), "AVERAGE"
		print reduce(lambda x, y: x + y, recalls) / len(recalls), "RECALLS"
			

	def test(self, debug=False):
		""" Runs through testing data once, returns accuracy and recall """
		count = 0
		correct = 0
		conf = {"tp":0, "fp":0, "tn":0, "fn":0}
		for dct in self.testing_data:
			count += 1
			res, vct = self.classify(dct['text'])
			if res == dct['label']:
				correct += 1
				if res == 1:
					conf["tp"] += 1
				else:
					conf["tn"] += 1
			else:
				if res == 1:
					conf["fp"] += 1
					if debug:
						print dct['text']
						print vct
						print "FALSE POSITIVE --------------------------------------"
				else:
					conf["fn"] += 1
					if debug:
						print dct['text']
						print vct
						print "FALSE NEGATIVE --------------------------------------"
		accuracy = (correct*100.0/count)
		recall = conf['tp']*100.0/(conf['tp']+conf['fn'])
		print "{}/{}={}%".format(correct, count, accuracy)
		print conf
		return accuracy, recall

	def set_training_testing_data(self, portion):
		""" Splits data into two parts depending on the portion parameter """
		if portion <= 0 or portion >= 1:
			portion = 0.9
		border_index = int(round(len(self.data)*portion))
		self.training_data = self.data[:border_index]
		self.testing_data = self.data[border_index:]

	def set_classifier(self):
		""" Converts data to feature vectors, trains the model. """
		formatted_data = self.get_feature_vectors()
		self.classifier = SklearnClassifier(LinearSVC()).train(formatted_data)

	def get_feature_vectors(self):
		""" Returns a list of feature vectors for training data """
		vector_lst = []
		for dct in self.training_data:
			features = self.extract_features(self.extractor_index, dct['text'])
			if features is not None:
				vector_lst.append((features, dct['label']))
		return vector_lst

	def classify(self, text):
		""" Classifies one tweet """
		vector = self.extract_features(self.extractor_index, text)
		return self.classifier.classify(vector), vector

	def extract_features(self, index, text):
		""" Returns a vector """
		if index == 1:
			# 65%
			# number of occurences for categories in the vocab are recorded
			return self.extract_vocab_structure(text)
		elif index == 2:
			# 88%
			# any word that doesn't fall into any known cat, is being recorded separately as a Boolean
			return self.extract_vocab_structure(text, record_unrecognized=True)
		elif index == 3:
			# 88%
			# same as above but names fall into mentions
			return self.extract_vocab_structure(text, namesToMentions=True, record_unrecognized=True)
		else:
			return None


	def extract_vocab_structure(self, text, namesToMentions=False, record_unrecognized=False):
		""" Does the same as the extractor 1 but saves the unrecognised words, too, as Booleans """
		try:
			# it's not sentiment analysis so we just need text
			cleaned, res = self.process_parsing(text.encode('ascii', 'ignore'))
			tokens = TweetTokenizer().tokenize(cleaned)	
			tokens = self.remove_stopwords(tokens)		
			res.update(self.tokens_to_vocab(tokens, namesToMentions=namesToMentions, record_unrecognized=record_unrecognized))
			return res	
		except (UnicodeDecodeError, UnicodeEncodeError) as e:
			print "Unicode Error"
			self.errors += 1
			return None

	def tokens_to_vocab(self, tokens, namesToMentions=False, record_unrecognized=False):
		""" Checks if every token falls into a vocab structure. Retuns a
		dict of a format {category:numOfOccurrences} """
		
		res = {}
		for t in tokens:
			done = False
			for key in self.vocab.keys():
				key_m = key
				if self.check_vocab(t, self.vocab[key_m], key_m):
					if key_m=='names' and namesToMentions:
						key_m = 'mentions'
					if not key_m in res:
						res[key_m] = 0
					res[key_m]+=1
					done = True
			if (not done) and record_unrecognized:
				res[t] = True
		return res 

	def check_vocab(self, token, wordlist, categoryName):
		""" Token is one word but word can be a concept consisting of 2 wds or 
		a concept with an underscore"""
		if categoryName == 'stems':
			if self.wnl.lemmatize(token.lower()) in wordlist:
				#print "found {} in {}".format(token, categoryName)
				return True
			return False
		for word in map(str.lower, wordlist):
			if " " in word:
				wds = re.split(' ', word)
				if token.lower() in map(str.lower, wds):
					#print "found {} in {}".format(token, categoryName)
					return True 
			else:
				if token.lower() == word.lower():
					#print "found {} in {}".format(token, categoryName)
					return True 
		return False

	def remove_stopwords(self, tokens):
		""" Removes all sorts of words that do not hold meaning """
		twitter_specific = ["RT"]
		tokens = [tok for tok in tokens if tok not in twitter_specific]
		tokens = [tok[:-2] if tok.endswith("'s") else tok for tok in tokens]
		return tokens
		stopwds = stopwords.words('english')
		punct =  list(string.punctuation)
		tokens = [tok for tok in tokens if tok not in punct]
		tokens = [tok for tok in tokens if tok not in stopwds]
		tokens = [tok.lower() if not tok.isupper() and not tok.islower() else tok for tok in tokens ]
		return tokens

	def process_parsing(self, text_str):
		""" Takes text, extracts mentions, checks if these are knows.
		Returns a dict of a format {mentions:numOfOccurrences} """
		res = {}
		parsed = preprocessor.parse(text_str)
		if parsed.mentions:
			# count how many mentions are known
			mention_list = [o.match.lower() for o in parsed.mentions]
			occurrences = len(filter(lambda mention: self.mention_known(mention), mention_list))
			if occurrences > 0:
				res['mentions'] = occurrences
		preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.EMOJI, preprocessor.OPT.MENTION)
		cleaned = preprocessor.clean(text_str)
		return cleaned, res

	def mention_known(self, mention):
		""" Checks if a mention is known """
		return mention in self.vocab['mentions']

if __name__ == '__main__':
	main()