#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess

# 10.0 0.616040955631
# 100. 0.616040955631
# 1000 0.821387940842
# 10000 0.892491467577
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
clf=SVC(kernel="rbf",C=10000.0)
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]
t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"
t0 = time()
print list(clf.predict(features_test))
# print accuracy_score(clf.predict(features_test),labels_test)
print "training time:", round(time()-t0, 3), "s"


#########################################################
### your code goes here ###

#########################################################


