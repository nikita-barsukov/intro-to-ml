#!/usr/bin/python

import sys
import pickle
sys.path.append("tools/")

from pprint import pprint
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','total_payments', 'total_stock_value', 'from_poi_to_this_person', 'to_messages',
                 'deferral_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'expenses', 'exercised_stock_options', 'other',
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
for i in my_dataset:
    new_feature = None
    try:
        new_feature = my_dataset[i]['bonus'] / my_dataset[i]['total_payments']
    except:
        new_feature = 0
    my_dataset[i]['bonus_share'] = new_feature

features_list.append('bonus_share')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

estimator_tree, estimator_naive, estimator_svm = ([('rescale', MinMaxScaler()),
                                                    ('select_features', SelectKBest(f_classif, k=10)),
                                                    ('reduce_dim', PCA())] for i in range(3))
estimator_tree.append(('tree', tree.DecisionTreeClassifier()))
estimator_naive.append(('naive', GaussianNB()))
estimator_svm.append(('svm', SVC(kernel="linear", C=1000)))
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
    pred = clf.predict(f_test)
    return accuracy_score(l_test, pred)

accuracies = map(lambda cl: get_accuracy(cl, f_train=features_train,
                                         f_test=features_test,
                                         l_train=labels_train,
                                         l_test=labels_test), classifiers)

clf = classifiers[accuracies.index(max(accuracies))]
scores = clf.named_steps['select_features'].scores_

features_selected_bool = clf.named_steps['select_features'].get_support(indices=True)
features_selected = [features_list[i+1] for i in features_selected_bool]
features_scores = [scores[i] for i in features_selected_bool ]
print('Selected features:')
print(features_selected)
print('Feature scores:')
for i in range(len(features_scores)):
    print features_selected[i], '{0:.10f}'.format(features_scores[i])

features_list.insert(0, 'poi')
pred = clf.predict(features_test)
print("")
print("Accuracy report:")
print(classification_report(labels_test, pred))
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_selected)
