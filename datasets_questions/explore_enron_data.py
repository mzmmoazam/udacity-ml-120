#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print [i for i in enron_data.keys()if "FASTOW" in i]
print (enron_data["SKILLING JEFFREY K"]["total_payments"])
print (enron_data["LAY KENNETH L"]["total_payments"])
print (enron_data["FASTOW ANDREW S"]["total_payments"])
c=k=0
for i in enron_data:
    if enron_data[i]["total_payments"]!="NaN":
        c+=1
        print enron_data[i]["salary"]
    k+=1
    # if enron_data[i]["email_address"]!='NaN':
    #     k+=1
    #     print enron_data[i]["email_address"]
print c,k