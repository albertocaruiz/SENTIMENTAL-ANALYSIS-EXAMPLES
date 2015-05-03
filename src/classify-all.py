import numpy as np
from random import shuffle
import sklearn
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation


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

no_iter = 1
avg_acc = []
while(no_iter > 0):
	no_iter -=1
	all_instances = readInput()
	shuffle(all_instances)
	# print all_instances[0:700]
	# print "all_instances size: ", len(all_instances)
	train_set = np.array(all_instances)

	X = np.array(train_set[:,:-1])
	Y = np.array(train_set[:,-1])
	Y = [binary(i) for i in Y]

	svc = svm.SVC()
	lr = LogisticRegression(penalty='l1', tol=0.01)
	gnb = GaussianNB()

	kfold = cross_validation.KFold(len(all_instances), n_folds=10)
	
	svc_accuracy = cross_validation.cross_val_score(svc, X, Y, cv=kfold)
	lr_accuracy = cross_validation.cross_val_score(lr, X, Y, cv=kfold)
	gnb_accuracy = cross_validation.cross_val_score(gnb, X, Y, cv=kfold)


	print 'SVM average accuary: %f' %svc_accuracy.mean()
	print 'LogisticRegression average accuary: %f' %lr_accuracy.mean()
	print 'Naive Bayes average accuary: %f' %gnb_accuracy.mean()





