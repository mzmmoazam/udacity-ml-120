#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
# print [data_dict[i] for i in data_dict]
data_dict.pop("TOTAL",0)
print [[q,data_dict[q]["salary"],data_dict[q]["bonus"]] for q in data_dict if data_dict[q]["salary"] >= 1000000 and data_dict[q]["bonus"] >= 5000000 and data_dict[q]["bonus"]!='NaN']
features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

m=0
### your code below
for point in data:
    # print point
    m=point[0]if m<point[0] and point[0]!=26704229.0 else m
    salary = point[0]
    bonus = point[1]
    matplotlib.pyplot.scatter( salary, bonus )
print m
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()


