from BeautifulSoup import BeautifulSoup 
import requests 
import json
import os

class EmojiSentiment():

	def __init__(self, ):
		if os.path.isfile("/cs/home/mn39/Documents/MSciDissertation/resources/emoji_sentiment.txt"):
			self.sentiment = self.get_data_locally()
		else:
			self.sentiment = self.get_data_online()
			self.write_out()

	def get_sentiment(self, character):
		if character in self.sentiment:
			return self.sentiment[character]
		print "cannot find {}".format(character)
		return None

	def write_out(self):
		if self.sentiment:
			json.dump(self.sentiment, open("/cs/home/mn39/Documents/MSciDissertation/resources/emoji_sentiment.txt",'w'))

	def get_data_locally(self):
		return json.load(open("/cs/home/mn39/Documents/MSciDissertation/resources/emoji_sentiment.txt"))

	def get_data_online(self):
		res = {}
		url = "http://kt.ijs.si/data/Emoji_sentiment_ranking/index.html"
		r = requests.get(url)
		soup = BeautifulSoup(r.text)
		table = soup.find('table', { "id" : "myTable" })
		rows = table.findAll("tr")
		rows = rows[1:]
		for row in rows:
			cells = row.findAll("td")
			code = str(cells[2].text)
			score = float(cells[8].text) 
			res[code] = score 
		return res

if __name__ == '__main__':
	es = EmojiSentiment()
	r1 = es.get_sentiment("0x2764")
	es.write_out()
	es.get_data_locally()
	print r1 == es.get_sentiment("0x2764")

