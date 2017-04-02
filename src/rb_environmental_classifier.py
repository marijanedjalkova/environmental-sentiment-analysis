import preprocessor 
from nltk.tokenize import TweetTokenizer
import re

MENTIONS = ['falkirkcouncil', '^INEOS(_([^\s]+))?$', '^BP(_([^\s]+))?$', 'ICI', 'ScottishEPA']

NAMES = ['ineos', 'petroineos', 'bp', 'calachem', 'ici', 'falkirk council', 'sepa']

NOUNS = ['smoke', 'flare', 'smog', 'fume', 'dust', 'asthma', 'idling', 'burn', 'fish', 'egg', 
'smell', 'odour', 'fly', 'rodent', 'vermin', 'insect', 'pest', 'rat', 'mouse', 'cockroach',
'dirt', 'rubbish', 'dump', 'landfill', 'litter', 'fly-tip', 'disgust', 'danger', 'health',
'harm', 'pollution', 'oil', 'sewage', 'hydrocarbons', 'flood', 'environment']

VERBS = ['smell', 'stink', 'reek']

ADJECTIVES = ['gross', 'harmful', 'horrible', 'nasty', 'vile', 'foul']

BINGO = 1
MISS = 0

def is_in_mentions(mention):
	for m in MENTIONS:
		if re.match(m, mention):
			return True
	return False

def is_in_words(word):
	if word in NOUNS:
		return True 
	if word in VERBS:
		return True 
	if word in ADJECTIVES:
		return True
	if check_other_forms(word):
		return True 
	return False

def check_other_forms(word):
	# check other forms of the same word

def is_is_names(word):
	return word in NAMES


class RB_classifier(object):

	def __init__(self):
		self.lexicon = self._initialise_lexicon() 

	def _initialise_lexicon(self):
		return {}
		# fill it with other forms of the words in given list 

	def classify(self, text):
		# decode
		text = text.encode('utf8')
		# tokenize 
		some_result = self.parsing(text)
		tokens = preprocessor.tokenize()
		# get rid of garbage 
		# analyse 
		return res

	def parsing(self, text):
		try:
			parsed = preprocessor.parse(text)
			if parsed.urls:
				 pass
			if parsed.mentions:
				for m in parsed.mentions:
					if is_in_mentions(m.match):
						pass
						# well, this should increase the chances by a loooooooot
			if parsed.emojis:
				pass
		except (UnicodeDecodeError, UnicodeEncodeError) as e:
			return 0

if __name__ == '__main__':
	print is_in_mentions("INEOS")
	print is_in_mentions("INEOS_abc")
	print is_in_mentions("INEOS_")
	print is_in_mentions("blaneos")
	print is_in_mentions("ici")
	print is_in_mentions("ICI")
	print is_in_mentions("BP")
	print is_in_mentions("BP_abc_efd")
