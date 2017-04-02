import urllib2
import json

def get_single_polarity(tweet_text, neg_value, pos_value):
	""" Will return positive is not negative. """
	data = {"data": [{"text": ""}]}
	new_list = []
	for it in tweet_text:
		if it.startswith("#"):
			new_list.append(it[1:])
		else:
			new_list.append(it)
	joined = "+".join(new_list) 
	data["data"][0]["text"] = joined
	req = 'http://www.sentiment140.com/api/classify?text=' + joined
	try:
		response = urllib2.urlopen(req) 
		page = response.read()
		d = json.loads(page)
	except:
		return -2
	p = d["results"]["polarity"]
	if p == 0: 
		return neg_value
	if p == 4:
		return pos_value
	# return something in the middle, can then compare in a more complicated way
	#return (neg_value + pos_value) * 1.0 / 2
	return pos_value


if __name__ == '__main__':
	print get_single_polarity("woohoo!!! lol i love this", 0 , 1)
	

	