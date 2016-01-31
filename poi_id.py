#!/usr/bin/python

import sys
import pickle
sys.path.append("tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from ggplot import *
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','total_payments', 'total_stock_value', 'from_poi_to_this_person', 'to_messages',
                 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
                 'long_term_incentive', 'restricted_stock',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
del(data_dict['TOTAL'])

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

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
from sklearn import tree
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif

estimator_tree, estimator_naive, estimator_svm  = ([('rescale', MinMaxScaler()),
                                                    ('select_features', SelectKBest(f_classif, k=10)),
                                                    ('reduce_dim', PCA())] for i in range(3))
estimator_tree.append(('tree', tree.DecisionTreeClassifier()))
estimator_naive.append(('naive', GaussianNB()))
estimator_svm.append(('svm', SVC(kernel="rbf", C=1000)))
classifiers = [
    Pipeline(estimator_naive),
    Pipeline(estimator_tree),
    Pipeline(estimator_svm)
]

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

def get_accuracy(clf, f_train, f_test, l_train, l_test):
    clf.fit(f_train, l_train)
    from sklearn.metrics import accuracy_score
    pred = clf.predict(f_test)
    return accuracy_score(l_test, pred)

accuracies = map(lambda cl: get_accuracy(cl, f_train=features_train,
                              f_test=features_test,
                              l_train=labels_train,
                              l_test=labels_test), classifiers)

clf = classifiers[accuracies.index(max(accuracies))]
features_selected_bool = clf.named_steps['select_features'].get_support()
features_list = [x for x, y in zip(features_list[1:], features_selected_bool) if y]
features_list.insert(0, 'poi')
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
