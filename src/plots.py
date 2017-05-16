import matplotlib.pyplot as plt
import numpy as np
import csv
import matplotlib.cbook as cbook

def extract_data(data_from_csv, classifier_name, field_name):
	acc = {'1':[], '2':[], '3':[]}

	for i in range(0, len(data_from_csv['n'])):
		if data_from_csv['Classifier'][i]==classifier_name:
			n = data_from_csv['n'][i]
			acc[n].append(data_from_csv[field_name][i])
	return acc

def read_data(fname):
	data_from_csv = {}
	with open(fname, 'rb') as csvfile:
		reader = csv.reader(csvfile)
		headers = reader.next()
		for h in headers:
			data_from_csv[h] = []
		for row in reader:
			for h, v in zip(headers, row):
				data_from_csv[h].append(v)
	return data_from_csv


def get_col(data_from_csv, col):
	res_set = []
	for t in data_from_csv[col]:
		if t not in res_set:
			res_set.append(t)
	return res_set

def plot_experiment(classifier, x_axis_name, y_axis_name, x_label, y_label, fname):
	data_from_csv = read_data('/cs/home/mn39/Documents/MSciDissertation/src/sentimentvalidationoutput.csv')
	data = extract_data(data_from_csv, classifier, y_axis_name)
	x_axis = get_col(data_from_csv, x_axis_name)

	fig, ax1 = plt.subplots()
	line1, = ax1.plot(x_axis, data['1'])
	line2, = ax1.plot(x_axis, data['2'])
	line3, = ax1.plot(x_axis, data['3'])
	ax1.set_ylabel(y_label)
	ax1.set_xlabel(x_label)

	l_parts = [line1, line2, line3]
	plt.legend(l_parts, ('unigrams', 'bigrams', 'trigrams'), loc='best')

	plt.savefig(fname)

def compare_all():
	classifiers = ['SVM', "NaiveBayes"]
	ys = ["accuracy", "recall"]
	for c in classifiers:
		for y in ys:
			fname = "/cs/home/mn39/Documents/MSciDissertation/docs/{}{}.png".format(c, y[:3])
			plot_experiment(c, "training", y, "Training set size", y, fname)

def cmp_class_unigrams(field):
	data_from_csv = read_data('/cs/home/mn39/Documents/MSciDissertation/src/sentimentvalidationoutput.csv')
	SVMdata = extract_data(data_from_csv, "SVM", field)['1']
	x_axis = get_col(data_from_csv, 'training')
	NBdata = extract_data(data_from_csv, "NaiveBayes", field)['1']
	fig, ax1 = plt.subplots()
	line1, = ax1.plot(x_axis, SVMdata)
	line2, = ax1.plot(x_axis, NBdata)
	ax1.set_ylabel(field)
	ax1.set_xlabel("Training data")
	l_parts = [line1, line2]
	plt.legend(l_parts, ('SVM unigrams', 'Naive Bayes unigrams'), loc='best')
	plt.savefig("/cs/home/mn39/Documents/MSciDissertation/docs/class{}Comp.png".format(field[:3].upper()))

def autolabel(rects, ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.01*height,
                '%d' % int(height),
                ha='center', va='bottom')

def topic_experiments():
	d = read_data('/cs/home/mn39/Documents/MSciDissertation/src/topickfoldoutput.csv')
	recs = [float(i) for i in d['recall']]
	accs = [float(i) for i in d['accuracy']]
	x_axis = np.array(range(0,4))
	fig, ax1 = plt.subplots()
	bar_width = 0.35
	rects1 = ax1.bar(x_axis, accs, bar_width, label='Accuracy')
	rects2 = ax1.bar(x_axis + bar_width, recs, bar_width, label='Recall')
	plt.xticks(x_axis + bar_width / 2, ('Extractor 1', 'Extractor 2', 'Extractor 3', 'Extractor 4'))
	plt.legend(loc='lower right')
	plt.tight_layout()
	autolabel(rects1, ax1)
	autolabel(rects2, ax1)
	plt.savefig("/cs/home/mn39/Documents/MSciDissertation/docs/topic.png")

def main():
	#cmp_class_unigrams('accuracy')
	#cmp_class_unigrams('recall')
	topic_experiments()

if __name__ == '__main__':
	main()
