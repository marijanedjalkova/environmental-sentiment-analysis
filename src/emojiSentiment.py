from BeautifulSoup import BeautifulSoup 
import requests 
import json

class EmojiSentiment():

	def __init__(self, ):
		self.sentiment = self.get_data_online()

	def get_sentiment(self, character):
		return self.sentiment[character]

	def write_out(self):
		if self.sentiment:
			json.dump(self.sentiment, open("../resources/emoji_sentiment.txt",'w'))

	def get_data_locally(self):
		self.sentiment = json.load(open("../resources/emoji_sentiment.txt"))

	def get_data_online(self):
		res = {}
		url = "http://kt.ijs.si/data/Emoji_sentiment_ranking/index.html"
		r = requests.get(url)
		data = r.text 
		data = data.replace("\t", "").replace("\r", "").replace("\n", "")
		soup = BeautifulSoup(data)
		#table = soup.find(lambda tag: tag.name=='table' and tag.has_key('id') and tag['id']=="myTable")
		#table = soup.find(lambda tag: tag.name=='table')
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

