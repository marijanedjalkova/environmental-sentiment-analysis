import csv

entities = csv.reader(open('/cs/home/mn39/Documents/MSciDissertation/resources/relatedEntities.csv'), delimiter = ',')
filtered = [entity for entity in entities if len(entity)>1]
cleaned = [name for [name,_] in filtered]
print cleaned