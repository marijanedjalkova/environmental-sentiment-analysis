import tweepy

class StreamListener(tweepy.StreamListener):

	def set_twitter_tool(self, twitter_tool):
		self.twitter_tool = twitter_tool

	def on_status(self, status):
		loc = status.user.location
		text = status.text
		coords = status.coordinates
		name = status.user.screen_name
		id_str = status.id_str
		created = status.created_at
		retweets = status.retweet_count
		# print text
		# print loc
		# print coords
		# print created
		print "----------------------------"
		# if text is necessary, send to Grangemouth
		if self.twitter_tool.fits(status):
			self.redirect_tweet(status)
		
	def on_error(self, status_code):
		if status_code == 420:
			return False

	def redirect_tweet(self, tweet):
		print "Sending the tweet is still to be implemented"