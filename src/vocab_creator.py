import csv


def read_csv_into_list(filename):
	entities = csv.reader(open(filename), delimiter = ',')
	filtered = [entity for entity in entities if len(entity)>1]
	cleaned = [name for [name,_] in filtered]
	print cleaned

def main():
	l1 = read_csv_into_list('/cs/home/mn39/Documents/MSciDissertation/resources/relatedEntities.csv')

if __name__ == '__main__':
	main()