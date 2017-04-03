import preprocessor 
from nltk.tokenize import TweetTokenizer
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import re

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

BINGO = 1
MISS = 0

BIASED = set(['job', 'salary', 'rugby', 'championship'])


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
		has_ellipsis = self.has_ellipsis(text)
		text = text.encode('utf8')
		# parse to get out emojis, urls and mentions 
		res = self.parsing(text, has_ellipsis)
		print "after parsing: ", res
		#clean out and tokenize
		preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.EMOJI, preprocessor.OPT.MENTION)
		cleaned = preprocessor.clean(text)
		tokens = TweetTokenizer().tokenize(cleaned)

		word_res = self.check_words(tokens)
		print "word checking returned ", word_res
		# analyse 
		res += word_res
		print "overall result ", res
		return res

	def has_ellipsis(self, text):
		# u2026 is a horizontal ellipsis. It usually means a link to another website
		# i.e. not what we are looking for
		# inless it is a link to instagram or twitpic
		if u"\u2026" in text:
			return True
		return False


	def check_words(self, tokens):
		value = -1
		for t in tokens:
			if t in BIASED:
				# decrease by quite a lot
				value -= 0.4
				continue
			if self.is_in_lexicon(t):
				# increase by a little 
				value += 0.2
				continue 
		return value


	def parsing(self, text, has_ellipsis):
		res = 0
		try:
			parsed = preprocessor.parse(text)
			if parsed.urls:
				res += self.check_urls(parsed.urls, has_ellipsis)
			if parsed.mentions:
				res += self.check_names(parsed.mentions)
			if parsed.emojis:
				res += self.check_emojis(parsed.emojis)
			return res
		except (UnicodeDecodeError, UnicodeEncodeError) as e:
			return 0

	def check_urls(self, urls, has_ellipsis):
		res = 0
		# TODO
		return res

	def check_emojis(self, emojis):
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
	r.classify(u"John McNally from @INEOS and Peter Miller from @BP_plc meet in Grangemouth, announcing 199m deal for forties pipeline and Kinneil Terminal")
