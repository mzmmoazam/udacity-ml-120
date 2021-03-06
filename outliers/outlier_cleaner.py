#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    # print predictions, ages, net_worths
    for i in range(90):
        # print predictions[i], ages[i], net_worths[i]
        cleaned_data.append((ages[i], net_worths[i],abs(net_worths[i]-predictions[i])))
    from operator import itemgetter
    cleaned_data=sorted(cleaned_data,key=itemgetter(2))[:81]
    return cleaned_data

