import csv

class DatasetAnalysis:

        def analyse(self):

		with open("/cs/home/mn39/Documents/MSciDissertation/resources/Sentiment-Analysis-Dataset.csv") as csvfile:
			data = csv.reader(csvfile) # 1578615 
			next(data, None) # skip headers
                        pos = 0
                        neg = 0
                        s1 = 0
                        s2 = 0
                        for row in data:
                            if int(row[1])==1:
                                pos+=1
                            else: neg+=1
                            if row[2] =="Sentiment140":
                                s1+=1
                            else: 
                                s2+=1
                                print row[2], row[0]
                                print row[3]
                        print "pos", pos*1.0/1578615*100, "%"
                        print "neg", neg*1.0/1578615*100, "%"
                        print "Sentiment140", s1*1.0/1578615*100, "%"
                        print "Kaggle", s2*1.0/1578615*100, "%"


if __name__=="__main__":
    d = DatasetAnalysis()
    d.analyse()

