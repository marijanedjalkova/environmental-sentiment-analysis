import csv
import os
import glob


def read_csv_into_list(filename):
	""" Read the Google-provided files into list of concepts """
	rows = csv.reader(open(filename), delimiter = ',')
	filtered = [row for row in rows if len(row)==2]
	cleaned = [name for [name,_] in filtered]
	return cleaned

def read_all_sources(path):
	extension = 'csv'
	os.chdir(path)
	vocab = []
	source_files = [i for i in glob.glob('*.{}'.format(extension))]
	for s in source_files:
		vocab.extend(read_csv_into_list(s))
	return list(set(vocab))

def write_to_file(filename, data):
	with open(filename, 'w') as output:
		wr = csv.writer(output)
		wr.writerows([data])

def create_local_vocab():
	""" This should be done once. Then the resulting vocab should be modified manually."""
	concepts = read_all_sources('/cs/home/mn39/Documents/MSciDissertation/resources/vocab_sources')
	write_to_file('/cs/home/mn39/Documents/MSciDissertation/resources/vocab.txt', concepts)

def read_in_local_vocab(filename):
	

def main():
	create_local_vocab()

if __name__ == '__main__':
	main()