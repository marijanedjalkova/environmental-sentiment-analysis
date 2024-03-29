import preprocessor 
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import re
import urllib2
from urlparse import urlparse

MENTIONS = ['falkirkcouncil', '^INEOS(_([^\s]+))?$', '^BP(_([^\s]+))?$', 'ICI', 'ScottishEPA']
ANTI_MENTIONS = ['MadrasRugby', 'SCRUMMAGAZINE'] # if these people are mentioned, this reduces the chances of it being a useful tweet

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

ANTI_LEXICON = set(['job', 'salary', 'rugby', 'champion', 'sell', 'asset'])

class HeadRequest(urllib2.Request):
    def get_method(self): return "HEAD"

def get_real_url(url):
	""" In tweets, all links come in a t.co format. This method returns the actual link. """
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
		""" Says YES / NO (1/0) for whether the tweet is needed or no. """
		# decode
		print text
		res = self.check_special_characters(text)
		text = text.encode('utf8')
		# parse to get out emojis, urls and mentions 
		parsing_res = self.parsing(text)
		res += parsing_res

		#clean out and tokenize
		preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.EMOJI, preprocessor.OPT.MENTION)
		cleaned = preprocessor.clean(text)
		tokens = TweetTokenizer().tokenize(cleaned)
		word_res = self.check_words(tokens)
		# analyse 
		res += word_res

		other_meaning = self.check_other_meanings(tokens)
		res += other_meaning

		if res >= 0.1:
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
		if 'oil' in tokens and 'refinery' in tokens:
			value -= 0.25
		for t in tokens:
			if self.wnl.lemmatize(t) in ANTI_LEXICON:
				# decrease by quite a lot
				value -= 0.4
				continue
			if self.is_in_lexicon(t):
				# increase by a little 
				value += 0.25
				continue 
		return value

	def check_other_meanings(self, tokens):
		""" Checks idioms and other things like irony.
		Irony not implemented yet. """
		# check irony???
		res = 0
		if 'smell' in tokens: # smell a rat
			index = tokens.index('smell')
			if index < len(tokens)-2 and tokens[index+1] == 'a' and tokens[index+2] == 'rat':
				res -= 0.6
		if 'fire' in tokens: # under fire
			index = tokens.index('fire')
			if index > 0 and tokens[index-1] == 'under':
				res -= 0.4
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
			return res
		except (UnicodeDecodeError, UnicodeEncodeError) as e:
			return 0

	def check_urls(self, urls):
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
				res -= 0.7
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
	print r.classify(u"snowing here now but it's like acid grangemouth pollution snow so doubt it will lie.")
	print r.classify(u"you been inhaling the fumes fae the grangemouth refinery ya loonbox??")
	print r.classify(u"Grangemouth oil refinery in all its finery.")
	print r.classify(u"@INEOS_Shale @F_F_Ryedale Ineos hit with safety notice over accident prevention at Grangemouth... https://www.energyvoice.com/other-news/healthandsafety/128599/ineos-hit-safety-notice-accident-prevention-grangemouth/ ... via @energyvoicenews")
	print r.classify(u"Ineos hit with safety notice over accident prevention at Grangemouth... Will @Ineosupstream do better at #fracking?")
	print r.classify(u"Everyone's talking about snow, I'm in Grangemouth with my chemical tan and the snow never lays")
	print r.classify(u"Day 5 of #365days - sunrise over Grangemouth chemical complex this morning. https://www.instagram.com/p/BO5TdjjAUBO/ ")
	print r.classify(u"what is that horrible smell")
	print r.classify(u"what is that weird smell")


