class DataSet:

	def __init__(self, filename, polarity_index, tweet_index, positive_value, negative_value, skip_header = False):
		self.filename = filename
		self.positive_value = positive_value
		self.negative_value = negative_value
		self.polarity_index = polarity_index
		self.tweet_index = tweet_index
		self.skip_header = skip_header