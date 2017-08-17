#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from sklearn.metrics import precision_score,recall_score

from sklearn.cross_validation import train_test_split
# print data,labels
### it's all yours from here forward!
x_data,y_data,x_labels,y_labels=train_test_split(features,labels,test_size=0.3,random_state=42)
from sklearn.tree import DecisionTreeClassifier
print len(y_labels)
clf=DecisionTreeClassifier()
clf.fit(x_data,x_labels)
print clf.predict(y_data),y_labels
print precision_score(clf.predict(y_data),y_labels)
print recall_score(clf.predict(y_data),y_labels)

print recall_score( [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0],[0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1])