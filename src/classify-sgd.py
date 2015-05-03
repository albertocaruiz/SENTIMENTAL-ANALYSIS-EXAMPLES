import numpy as np
import random
from sklearn import linear_model



def evaluation(predication, ground_truth):
	corr = 0.0
	for i in range(len(predication)):
		if predication[i] == ground_truth[i]:
			corr+=1

	print "Accuracy: ", corr/len(predication)



myinput1 = open("../vector/positive.txt")
myinput2 = open("../vector/negative.txt")

all_instances =[]

for f1 in myinput1:
	f1 = f1[:-1]
	f1 = f1.split(',')
	f1 = [int(i) for i in f1]
	all_instances.append(f1)

for f2 in myinput2:
	f2 = f2[:-1]
	f2 = f2.split(',')
	f2 = [int(i) for i in f2]
	all_instances.append(f2)


random.shuffle(all_instances)
# print all_instances[0:700]
print "all_instances size: ", len(all_instances)
train_set = np.array(all_instances[0:700])
test_set = np.array(all_instances[701:])

print train_set[:,:-1]

X = np.array(train_set[:,:-1])
Y = np.array(train_set[:,-1])
clf = linear_model.SGDClassifier()
clf.fit(X, Y)
predication = clf.predict(test_set[:,:-1])

evaluation(predication, test_set[:,-1])




