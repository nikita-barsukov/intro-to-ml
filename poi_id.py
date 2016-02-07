#!/usr/bin/python

import sys
import pickle
sys.path.append("tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from ggplot import *

import pandas as pd

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'total_payments', 'total_stock_value', 'from_poi_to_this_person', 'to_messages',
                 'deferral_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'expenses', 'exercised_stock_options', 'other',
                 'long_term_incentive', 'restricted_stock',
                 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# Exploring original dataset
data_dict_pandas = pd.DataFrame(data_dict).transpose()
print('Exploring ENRON dataset:')
print('------------')
shp = data_dict_pandas.shape
print shp[0], 'observations and', shp[1], 'features'

print('')
print('Number of missing data by feature')
print((data_dict_pandas=='NaN').sum())
poi_counts = data_dict_pandas['poi'].value_counts()

print('')
print('Persons of interest in dataset')
print(poi_counts)

print('---------------')

### Task 2: Remove outliers
# plotting salary and bonus variables to see if there are outliers
print ggplot(aes(x='salary', y='bonus'), data=data_dict_pandas) + geom_point()
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

estimator_tree, estimator_naive, estimator_svm, estimator_randf = ([('rescale', MinMaxScaler()),
                                                    ('select_features', SelectKBest(f_classif, k=10)),
                                                    ('reduce_dim', PCA())] for i in range(4))
estimator_tree.append(('tree', tree.DecisionTreeClassifier()))
estimator_naive.append(('naive', GaussianNB()))
estimator_svm.append(('svm', SVC(kernel="linear", C=1000)))
estimator_randf.append(('rand_forest', RandomForestClassifier(random_state=31)))

classifiers = [
    Pipeline(estimator_naive),
    Pipeline(estimator_tree),
    Pipeline(estimator_svm),
    Pipeline(estimator_randf)
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

def get_accuracy(cls, f_train, f_test, l_train, l_test):
    cls.fit(f_train, l_train)
    pred = cls.predict(f_test)
    return f1_score(l_test, pred)

accuracies = map(lambda cl: get_accuracy(cl, f_train=features_train,
                                         f_test=features_test,
                                         l_train=labels_train,
                                         l_test=labels_test), classifiers)
print('')
print 'Selecting classification algorithm.'
print('F1 scores:')
for acc in range(len(accuracies)):
    print classifiers[acc].steps[3][0], '{0:.2f}'.format(accuracies[acc])
# Selecting classifier with max F1 score
clf = classifiers[accuracies.index(max(accuracies))]

print("")
print('Selected classifying algorithm:')
print(classifiers[accuracies.index(max(accuracies))].steps[3][0])
print("")
print('Tuning classifier: selecting optimal number of features')

no_of_features = list(range(2, len(features_list) - 1))

def get_f1scores(no_of_features, f_train, f_test, l_train, l_test):
    params = [('rescale', MinMaxScaler()),
               ('select_features', SelectKBest(f_classif, k=no_of_features)),
               ('reduce_dim', PCA()),
               ('naive', GaussianNB())]
    return get_accuracy(Pipeline(params), f_train, f_test, l_train, l_test)

scores = map(lambda k: get_f1scores(k, f_train=features_train,
                                    f_test=features_test,
                                    l_train=labels_train,
                                    l_test=labels_test), no_of_features)

print 'Features\tF1 score'
for i in range(len(no_of_features)):
    print no_of_features[i], '\t\t\t{0:.2f}'.format(scores[i])

opt_features_indices = [i for i, val in enumerate(scores) if val == max(scores)]
opt_features = no_of_features[max(opt_features_indices)]

print "Optimal number of features", opt_features
clf = Pipeline([('rescale', MinMaxScaler()),
                ('select_features', SelectKBest(f_classif, k=opt_features)),
                ('reduce_dim', PCA()),
                ('naive', GaussianNB())])
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print("")
print("Efficiency of selected algorithm:")
print 'F1 score:\t', '{0:.2f}'.format(f1_score(labels_test, pred))
print 'Accuracy:\t',  '{0:.2f}'.format(accuracy_score(labels_test, pred))
print 'Precision:\t', '{0:.2f}'.format(precision_score(labels_test, pred))
print 'Recall:\t',  '{0:.2f}'.format(recall_score(labels_test, pred))

scores = clf.named_steps['select_features'].scores_
features_selected_bool = clf.named_steps['select_features'].get_support(indices=True)
features_selected = [features_list[i+1] for i in features_selected_bool]
features_scores = [scores[i] for i in features_selected_bool]

print("")
print('Feature scores:')
for i in range(len(features_scores)):
    print features_selected[i], '{0:.2f}'.format(features_scores[i])

features_selected.insert(0, 'poi')

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_selected)
