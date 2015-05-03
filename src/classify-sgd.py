import numpy as np
import random
import os
from sklearn import linear_model



def evaluation(prediction, ground_truth):
	corr = 0.0
	for i in range(len(prediction)):
		if prediction[i] == ground_truth[i]:
			corr+=1

	print "Accuracy: ", corr/len(prediction)
	return corr/len(prediction)


def readInput():
	path = "../vector/"
	fileList = [path + 'complaint.txt', path+'negative.txt', path+'positive.txt', path+'praise.txt']
	# myinput1 = open("../vector/positive.txt")
	# myinput2 = open("../vector/negative.txt")
	all_instances =[]
	for files in fileList:
		myinput = open(files)
		for f1 in myinput:
			f1 = f1[:-1]
			f1 = f1.split(',')
			f1 = [int(i) for i in f1]
			if f1[0] == 0:
				continue
			all_instances.append(f1)

	return all_instances

def binary(i): # to ignore -1/-2 or 1/2
	if i > 0: 
		return 1
	else:
		return -1

no_iter = 50
avg_acc = []
while(no_iter > 0):
	no_iter -=1
	all_instances = readInput()
	random.shuffle(all_instances)
	# print all_instances[0:700]
	# print "all_instances size: ", len(all_instances)
	train_set = np.array(all_instances[0:len(all_instances)*7/10])
	test_set = np.array(all_instances[len(all_instances)*7/10+1:])


	X = np.array(train_set[:,:-1])
	Y = np.array(train_set[:,-1])
	Y = [binary(i) for i in Y]

	clf = linear_model.SGDClassifier()
	clf.fit(X, Y)
	prediction = clf.predict(test_set[:,:-1])
	prediction = [binary(i) for i in prediction]

	ground_truth =  [binary(i) for i in test_set[:,-1]] 
	avg_acc.append(evaluation(prediction, ground_truth))

print "avg_acc: "
print sum(avg_acc)/len(avg_acc)



