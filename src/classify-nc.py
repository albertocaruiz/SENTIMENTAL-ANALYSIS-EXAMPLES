from sklearn.neighbors.nearest_centroid import NearestCentroid
import random
import numpy as np

def evaluation(predication, ground_truth):
	corr = 0.0
	for i in range(len(predication)):
		if predication[i] == ground_truth[i]:
			corr+=1

	print "Accuracy: ", corr/len(predication)
 
negative = open("/Users/sophiayan/Documents/cs410_GroupProject/SentimentalAnalysis/vector/positive.txt","r");
positive = open("/Users/sophiayan/Documents/cs410_GroupProject/SentimentalAnalysis/vector/negative.txt","r");

all_instances = [];

for row1 in positive:
	row1 = row1[:-1]
	row1 = row1.split(',')
	row1 = [int(i) for i in row1]
	all_instances.append(row1)

for row2 in negative:
	row2 = row2[:-1]
	row2 = row2.split(',')
	row2 = [int(i) for i in row2]
	all_instances.append(row2)
	
random.shuffle(all_instances)
# print all_instances[0:700]
print "all_instances size: ", len(all_instances)
train_set = np.array(all_instances[0:700])
test_set = np.array(all_instances[701:])

print train_set[:,:-1]

X = np.array(train_set[:,:-1])
Y = np.array(train_set[:,-1])

clf = NearestCentroid()
clf.fit(X, Y)
predication = clf.predict(test_set[:,:-1])

evaluation(predication, test_set[:,-1])