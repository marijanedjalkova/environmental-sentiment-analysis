from twitter_tool import * 
import json



def main():
	read_from_file()

def read_from_file():
	with open('elections.txt') as json_data:
		d = json.load(json_data)
	print(len(d))

def get_save_tweets():
	tt = TwitterTool()
	tweets = tt.search_tweets("election", 1500)
	tweet_list = tt.extract_text_from_tweets(tweets)
	data = []
	for tw in set(tweet_list):
		dct = {}
		dct["label"] = 0
		dct["text"] = tw
		data.append(dct)
	with open('elections2.txt', 'w') as outfile:
		json.dump(data, outfile)
	

if __name__ == '__main__':
	main()