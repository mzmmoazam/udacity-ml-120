#!/usr/bin/python

import sys
import pickle
import time

t0 =time.time()
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','shared_receipt_with_poi','defered_inc_ratio','defered_stk_ratio','msq_2_poi_rtio','msq_from_poi_rtio',] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
data_dict.pop("TOTAL",0)

### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
def if_NaN(feature):
    if feature =='NaN':
        return 0.0
    else:
        return feature

sys.path.append( "../tools/" )
import os
from parse_out_email_text import parseOutText

word_data=[]
def handling_incoming_emails(pickled=False):
    # if pickled:
    #     word_data=pickle.load(open('your_word_data.pkl'))
    #     for row in data_dict:
    #         for i in word_data:
    #             data_dict[row]['email_text'] =  i
    #             break
    #     return
    for row in data_dict:
        email_addr = data_dict[row]['email_address']
        emails=''
        try:
            emails = open('emails_by_address/from_' + email_addr + '.txt')
        except IOError as e:
            # print e
            data_dict[row]['email_text'] = ''

            continue
        for path in emails:
            if True:
                new_path = os.path.join('..', path[:-1][len('enron_mail_20110402')+1:])
                print new_path
                email = open(new_path, "r")
                temp_text = parseOutText(email)
                #### see to it later

                for i in ['attach','corpor','transport']:
                    if i in temp_text:
                        temp_text = temp_text.replace(i, '')

                ### append the text to word_data
                ### to store text to reduce time
                word_data.append(temp_text)
                data_dict[row]['email_text'] = temp_text
                email.close()

def sal():

    for row in data_dict:


        if data_dict[row]['total_payments']=='NaN' or data_dict[row]['deferred_income']=='NaN' :
            data_dict[row]['defered_inc_ratio']=0.0
        else:
            data_dict[row]['defered_inc_ratio'] = (data_dict[row]['total_payments'] + abs(data_dict[row]['deferred_income']))**2/abs(data_dict[row]['deferred_income'])


        if data_dict[row]['total_stock_value']=='NaN' or data_dict[row]['restricted_stock_deferred']=='NaN' :
            data_dict[row]['defered_stk_ratio']=0.0
        else:
            data_dict[row]['defered_stk_ratio'] = (data_dict[row]['total_stock_value'] + abs(data_dict[row]['restricted_stock_deferred']))**2/abs(data_dict[row]['restricted_stock_deferred'])

        if data_dict[row]['to_messages']=='NaN' or data_dict[row]['from_poi_to_this_person']=='NaN' :
            data_dict[row]['msq_2_poi_rtio']=0.0
        else:
            data_dict[row]['msq_2_poi_rtio'] = data_dict[row]['from_poi_to_this_person']/data_dict[row]['to_messages']

        if data_dict[row]['from_messages'] == 'NaN' or data_dict[row]['from_this_person_to_poi'] == 'NaN':
            data_dict[row]['msq_from_poi_rtio'] = 0.0
        else:
            data_dict[row]['msq_from_poi_rtio'] = data_dict[row]['from_this_person_to_poi'] / data_dict[row]['from_messages']


# print data_dict['METTS MARK'].keys(),data_dict['METTS MARK']

sal()
# handling_incoming_emails()


# print word_data,from_data
# store email_data here
# pickle.dump( word_data, open("your_word_data.pkl", "w") )
# words_file = "../text_learning/your_word_data.pkl"

# visualise feature to analyse for outliers and come up with new features
# import matplotlib.pyplot as plt
# for i in data_dict:
#     plt.scatter(data_dict[i]['poi'],data_dict[i]['defered_inc_ratio'])
# plt.xlabel("poi")
# plt.ylabel("defered_stk_ratio")
# plt.show()
###

# features_list.append('email_text')
my_dataset = data_dict
import numpy
numpy.random.seed(99)
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features = numpy.array(features,dtype=object)
# for i in features:
#     print i[5]

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
# clf = GaussianNB()
# clf = DecisionTreeClassifier()


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


# from sklearn.feature_extraction.text import TfidfVectorizer
#
# vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
#                              stop_words='english')
# features_train_email_vec = vectorizer.fit_transform([i[5]for i in features_train])
# features_test_email_vec = vectorizer.transform([i[5]for i in features_test]).toarray()
# print numpy.array(features_train_email_vec.toarray()).tolist(),numpy.array(features_test_email_vec.tolist())
#
# for i,j,i_,j_ in features_train,features_test,numpy.array(features_train_email_vec.toarray()).tolist(),numpy.array(features_test_email_vec.tolist()):
#     print i[5],j[5]
#     i[5],j[5] = i_,j_

# for i in range(len(features_train_email_vec.toarray())):
#         features_train[i][5] = features_train_email_vec.toarray()[i].tolist()
#
# for i in range(len(features_test_email_vec)):
#         features_test[i][5] = numpy.array(features_test_email_vec[i])

# features_train= numpy.array(features_train,dtype=numpy.float64)
# features_test= numpy.array(features_test,dtype=numpy.float64)

# for i in range(len(features_test_email_vec.toarray())):
#     print features_train[i][5]
#     features_test[i][5] = features_test_email_vec.toarray()[i]
# print features_train
# clf.fit(features_train_email_vec.toarray(),labels_train)
# xo=list(clf.feature_importances_)
# for i in xo:
    # print i
    # if i>0.15:
    #     print i,xo.index(i)
    #     print vectorizer.get_feature_names()[xo.index(i)]

# print "dfne",max(xo),vectorizer.get_feature_names()[327],vectorizer.get_feature_names()[1938    ]

# clf.fit(features_train,labels_train)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# PCA for decision tree
estimators = [('reduce_dim', PCA(n_components=5)), ('tree', DecisionTreeClassifier())]

# PCA for random forest
# estimators = [('reduce_dim', PCA(n_components=3)), ('rf',RandomForestClassifier() )]
clf = Pipeline(estimators)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# tree
from sklearn import grid_search

# parmaters of decision trees
parameters = {'min_samples_leaf': [1, 2, 3, 4, 5], 'min_samples_split': [2, 3, 4, 5]}

# parmaters of random forest
# parameters = {'min_samples_leaf':[1,2,3,4,5], 'min_samples_split':[2,3,4,5],
#              'n_estimators':[5,10,20,30]}
mclf = DecisionTreeClassifier()
# mclf = RandomForestClassifier()
clf = grid_search.GridSearchCV(mclf, parameters)
clf = clf.fit(features_train, labels_train)
clf = clf.best_estimator_
print clf
# clf = GaussianNB()

# NOTE, the result from GridSearch is a little different
# it does not give the best result in terms of precision and recall
# here i find this set parameters are better in terms of precision and recall
# clf = DecisionTreeClassifier(min_samples_leaf=1, min_samples_split=2)
# clf = RandomForestClassifier()



print clf.score(features_test,labels_test)
print  (time.time()-t0)/60


dump_classifier_and_data(clf, my_dataset, features_list)