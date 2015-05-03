import numpy as np
from random import shuffle
import sklearn
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

f = open("../vector/positive.txt")

f.readline()
data = np.loadtxt(f)

shuffle(data)
development = data[:4000,:] 
test = data[4000:,:] 

train = development[:,1:]
tag = development[:, 0] 

svc = svm.SVC(gamma=0.001, C=100.)

lr = LogisticRegression(penalty='l1', tol=0.01)

gnb = GaussianNB()
kfold = cross_validation.KFold(len(x1), n_folds=10)

svc_accuracy = cross_validation.cross_val_score(svc, train, tag, cv=kfold)
lr_accuracy = cross_validation.cross_val_score(lr, train, tag, cv=kfold)
gnb_accuracy = cross_validation.cross_val_score(gnb, train, tag, cv=kfold)


print 'SVM average accuary: %f' %svc_accuracy.mean()
print 'LogisticRegression average accuary: %f' %lr_accuracy.mean()
print 'Naive Bayes average accuary: %f' %gnb_accuracy.mean()