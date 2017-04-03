import preprocessor 
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import re
import urllib2
from urlparse import urlparse

MENTIONS = ['falkirkcouncil', '^INEOS(_([^\s]+))?$', '^BP(_([^\s]+))?$', 'ICI', 'ScottishEPA']
ANTI_MENTIONS = ['MadrasRugby'] # if these people are mentioned, thiw reduces the chances of it being a useful tweet

NAMES = ['ineos', 'petroineos', 'bp', 'calachem', 'ici', 'falkirk council', 'sepa']

NOUNS = set(['smoke', 'flare', 'smog', 'fume', 'dust', 'asthma', 'idling', 'burn', 'fish', 'egg', 
'smell', 'odour', 'fly', 'rodent', 'vermin', 'insect', 'pest', 'rat', 'mouse', 'cockroach',
'dirt', 'rubbish', 'dump', 'landfill', 'litter', 'fly-tip', 'disgust', 'danger', 'health',
'harm', 'pollution', 'oil', 'sewage', 'hydrocarbons', 'flood', 'environment'])

VERBS = set(['smell', 'stink', 'reek'])

ADJECTIVES = set(['gross', 'harmful', 'horrible', 'nasty', 'vile', 'foul'])
GIVEN_LEXICON = {'NOUNS': NOUNS, 'VERBS': VERBS, 'ADJECTIVES': ADJECTIVES}

YES = 1
NO = 0

BIASED = set(['job', 'salary', 'rugby', 'championship', 'sell', 'sells'])

class HeadRequest(urllib2.Request):
    def get_method(self): return "HEAD"

def get_real_url(url):
    res = urllib2.urlopen(HeadRequest(url))
    return res.geturl()


class RB_classifier(object):


	def __init__(self):
		self.wnl = WordNetLemmatizer()
		self.stem_lexicon = self._initialise_lexicon() 

	def _initialise_lexicon(self):
		""" Finds other forms of the same words and their synonyms and adds to the lexicon """
		
		stems = set()
		for pos in GIVEN_LEXICON:
			for w in GIVEN_LEXICON[pos]:
				stem = self.wnl.lemmatize(w)
				stems.add(stem)
				synonyms = self._get_synonyms(w)
				for s in synonyms:
					stem = self.wnl.lemmatize(s)
					stems.add(stem)
		return stems


	def _get_synonyms(self, word):
		""" Returns a list of synonyms of a word """
		syns = wordnet.synsets(word)
		return [l.name().encode('utf8') for s in syns for l in s.lemmas()]


	def classify(self, text):
		# decode
		res = check_special_characters(text)
		text = text.encode('utf8')
		# parse to get out emojis, urls and mentions 
		parsing_res = self.parsing(text)
		res += parsing_res
		#print "after parsing: ", res
		#clean out and tokenize
		preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.EMOJI, preprocessor.OPT.MENTION)
		cleaned = preprocessor.clean(text)
		tokens = TweetTokenizer().tokenize(cleaned)

		word_res = self.check_words(tokens)
		#print "word checking returned ", word_res
		# analyse 
		res += word_res
		other_meaning = self.check_other_meanings(tokens)
		res += other_meaning
		print res
		if res > 0:
			return YES
		return NO

	def check_special_characters(self, text):
		""" Special check for unicode characters """
		res = 0
		if u"\u2026" in text:
			# horizontal ellipsis -> news link
			res -= 0.5
		if u"\u0024" in text or u"\u20AC" in text or u"\u00A3" in text:
			# currency 
			res -= 0.6
		return res


	def check_words(self, tokens):
		value = 0
		for t in tokens:
			if t in BIASED:
				# decrease by quite a lot
				value -= 0.4
				continue
			if self.is_in_lexicon(t):
				# increase by a little 
				value += 0.3
				continue 
		return value

	def check_other_meanings(self, tokens):
		""" Checks idioms and other things like irony.
		Irony not implemented yet. """
		# check irony???
		res = 0
		if 'smell' in tokens:
			index = tokens.index('smell')
			if tokens[index+1] == 'a' and tokens[index+2] == 'rat':
				res -= 0.6
		return res


	def parsing(self, text):
		""" Uses preprocessor to extract features from tweets and analyze them """
		res = 0
		try:
			parsed = preprocessor.parse(text)
			if parsed.urls:
				res += self.check_urls(parsed.urls)
			if parsed.mentions:
				res += self.check_names(parsed.mentions)
			if parsed.emojis:
				res += self.check_emojis(parsed.emojis)
			return res
		except (UnicodeDecodeError, UnicodeEncodeError) as e:
			return 0

	def check_urls(self, urls, has_ellipsis):
		""" Processes urls from a tweet """
		res = 0
		for url in urls:
			res -= 0.5
			try:
				real_url = get_real_url(url.match)
				domain = urlparse(real_url).netloc
				if domain not in ['www.instagram.com', 'twitter.com']:
					res -= 0.25
			except urllib2.HTTPError:
				continue
		return res

	def check_emojis(self, emojis):
		""" Processes emojis from a tweet. Not implemented yet. """
		res = 0
		for em in emojis:
			print em.match 
		return res

	def check_names(self, mentions):
		""" Performs checks on all the names in a tweet """
		res = 0
		for m in mentions:
			name = m.match[1:]
			if self.is_in_mentions(name):
				res += 0.5
				# should increase the chances by a lot
				# but not too much, I guess
			elif name in ANTI_MENTIONS:
				res -= 0.5
		return res
						

	def is_in_mentions(self, mention):
		""" Checks if the mention is of someone of interest. """
		for m in MENTIONS:
			if re.match(m, mention):
				return True
		if mention in NAMES:
			return True
		return False


	def is_in_lexicon(self, word):
		""" Checks if the word has been given initially or shares the stem with a  """
		if word in NOUNS:
			return True 
		if word in VERBS:
			return True 
		if word in ADJECTIVES:
			return True
		if self.check_lexicon(word):
			return True 
		return False

	def check_lexicon(self, word):
		""" Gets stem of the word and checks if it is in the lexicon """
		return self.wnl.lemmatize(word) in self.stem_lexicon 

if __name__ == '__main__':
	r = RB_classifier()
	print r.classify(u"why does it always stink in Grangemouth")
