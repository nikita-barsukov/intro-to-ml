This is the file where I track improvement of my classifier.

1. Initial code.

```
	Accuracy: 0.25560	Precision: 0.18481	Recall: 0.79800	F1: 0.30011	F2: 0.47968
	Total predictions: 10000
	True positives:  1596
	False positives: 7040
	False negatives:  404
	True negatives:   960
```

2. Total payments feature instead of salary
```
	Accuracy: 0.76308	Precision: 0.06938	Recall: 0.04350	F1: 0.05347	F2: 0.04701
	Total predictions: 13000
	True positives:   87
	False positives: 1167
	False negatives: 1913
	True negatives: 9833
```

3. Removed outlier
```
	Accuracy: 0.82600	Precision: 0.10542	Recall: 0.01750	F1: 0.03002	F2: 0.02100
	Total predictions: 13000
	True positives:   3
	False positives:  297
	False negatives: 1965
	True negatives: 10703
```

4) Feature scaling, more features

```
	Accuracy: 0.83000	Precision: 0.11382	Recall: 0.02800	F1: 0.04494	F2: 0.03297
	Total predictions: 14000
	True positives:   56
	False positives:  436
	False negatives: 1944
	True negatives: 11564
```

5) Decision tree not Naive Bayes
```
	Accuracy: 0.85207	Precision: 0.39650	Recall: 0.06800	F1: 0.11609	F2: 0.08151
	Total predictions: 14000
	True positives:  136
	False positives:  207
	False negatives: 1864
	True negatives: 11793
```

6) Pipeline, all features with min max scaler
```
	Accuracy: 0.80120	Precision: 0.27311	Recall: 0.29550	F1: 0.28386	F2: 0.29073
	Total predictions: 15000
	True positives:  591
	False positives: 1573
	False negatives: 1409
	True negatives: 11427

7) Select 10 features, trying multiple classifiers
```
	Accuracy: 0.83573	Precision: 0.38090	Recall: 0.37100	F1: 0.37589	F2: 0.37294
	Total predictions: 15000
	True positives:  742
	False positives: 1206
	False negatives: 1258
	True negatives: 11794
```

8) Select K best features, final algorithm classifier
```
	Accuracy: 0.84360	Precision: 0.40137	Recall: 0.35200	F1: 0.37507	F2: 0.36088
	Total predictions: 15000	
	True positives:  704	
	False positives: 1050	
	False negatives: 1296	
	True negatives: 11950
```
