import csv
import os
import glob


def read_csv_into_list(filename):
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


def main():
	l = read_all_sources('/cs/home/mn39/Documents/MSciDissertation/resources/vocab_sources')
	print len(l)

if __name__ == '__main__':
	main()