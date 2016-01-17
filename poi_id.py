#!/usr/bin/python

import sys
import pickle
sys.path.append("tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from ggplot import *
import pandas as pd

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','total_payments', 'from_poi_to_this_person', 'to_messages']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
# plt_all = ggplot(pd.DataFrame(data_dict).transpose(), aes('salary', 'bonus')) + geom_point()
del(data_dict['TOTAL'])
# plt_no_outliers = ggplot(pd.DataFrame(data_dict).transpose(), aes('salary', 'bonus')) + geom_point()

# print(plt_all)
# print(plt_no_outliers)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict
def scale_feature(x, xmin, xmax):
    if x == 'NaN':
        return 0
    else:
        return (float(x) - xmin) / (xmax - xmin)

for feature in features_list:
    xmin = min(float(my_dataset[datapoint][feature]) for datapoint in my_dataset)
    xmax = max(float(my_dataset[datapoint][feature]) for datapoint in my_dataset)
    for datapoint in my_dataset:
        my_dataset[datapoint][feature] = scale_feature(my_dataset[datapoint][feature], xmin, xmax)


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)

labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

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

dump_classifier_and_data(clf, my_dataset, features_list)