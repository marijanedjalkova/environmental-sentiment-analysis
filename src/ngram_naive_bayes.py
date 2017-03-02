from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer
import csv
from textblob.classifiers import NaiveBayesClassifier
import re
from nltk.corpus import stopwords
import string
from nltk import ngrams

class Bigram_NBClassifier:

	def __init__(self, n, train_length, test_length):
                self.n = n
		self.dataset_len = 1578615
		self.train_limit = train_length
		self.test_limit = train_length + test_length
                self.tokenizer = TweetTokenizer()
                self.url_pattern = re.compile("(?P<url>https?://[^\s]+)")
                punctuation = list(string.punctuation)
                self.stopwds = stopwords.words('english') + punctuation + ['rt', 'via']
                self.train()

        def preprocess_tweet(self, text):
            decoded = text.decode("utf-8")
            tokens = self.tokenizer.tokenize(decoded)
            for index in range(len(tokens)):
                tok = tokens[index]
                if tok.startswith('@') and len(tok)>1:#not the 'i am @ the bar' cases
                    tokens[index] = "[MENTION]"
                if self.url_pattern.match(tok):
                    tokens[index]="[URL]"
            tokens = [tok for tok in tokens if tok not in self.stopwds]
            return tokens

        def ngram_extractor(self, document):
            features = {}
            for w in ngrams(document, self.n):
                features[w]=True
            return features

	def train(self):
		training_data = []
		with open("/cs/home/mn39/Documents/MSciDissertation/resources/Sentiment-Analysis-Dataset.csv") as csvfile:
			data = csv.reader(csvfile) # 1578615 
			next(data, None) # skip headers
			# format ItemID, Sentiment, SentimentSource, SentimentText
			for row in data:
			        source = row[2]
                                if source == "Sentiment140":
				        index = int(row[0])
				        if index % 10000 == 0:
					        print index
				        polarity = int(row[1]) # 0 or 1
				        if index < self.train_limit:
				                training_data.append((self.preprocess_tweet(row[3]), polarity))
                                        else:
                                                self.cl = NaiveBayesClassifier(training_data, feature_extractor = self.ngram_extractor)
					        break
			print "trained"

	def test(self):
                self.cl.show_informative_features(15)
		with open("/cs/home/mn39/Documents/MSciDissertation/resources/Sentiment-Analysis-Dataset.csv") as csvfile:
			data = csv.reader(csvfile) # 1578615 
			next(data, None) # skip headers
			correct = 0
			count = 0
                        kaggle = 0
			for row in data:
				source=row[2]
                                if source == "Sentiment140":
				        index = int(row[0])
				        if index % 10000 == 0:
					        print index
                                        tokens = self.preprocess_tweet(row[3])
				        polarity = int(row[1])
				        if self.train_limit < index < self.test_limit:
					        count +=1
					        predicted = self.cl.classify(tokens)
					        if predicted == polarity:
						        correct += 1
					        continue
				        elif index <= self.train_limit:
					        continue
				        else:
					        accuracy = correct * 1.0/(self.test_limit - self.train_limit - kaggle) 
					        print accuracy * 100 , "%"
					        return accuracy
                                else:
                                    kaggle += 1


if __name__=="__main__":
        nb = Bigram_NBClassifier(1, 100000, 1000)
        nb.test()

