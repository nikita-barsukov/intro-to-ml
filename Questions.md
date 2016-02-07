Enron Submission Free-Response Questions
==============


**1) Summarize the goal of this project and how machine learning is useful in trying to accomplish it.**

This project is a grading project for Udacity course "Introduction to Machine Learning". It deals with a dataset released in connection to famous Enron fraud scandal. Dataset contains information about key figures in Enron company at the time of that scandal, such as various compensation figures, data from email correspondence, etc. Goal of the project is to predict whether a person from dataset was convicted after fraud scandal or not using machine learning techniques.

**2) What features did you end up using in your POI identifier, and what selection process did you use to pick them?** 

In the beginning I used intuition to select features for my classifier. I started with 'total_payments' and 'from_poi_to_this_person', which produced reasonable accuracy but abysmal recall and precision. Then I decided to include all the features from the dataset. Additionally I added my own feature, share of bonus to total payments. 

Since features had vastly different scale, I implemented a min-max scaler using sklearn functions, with default parameters.  To select best features I used `SelectKBest()` function from sklearn package. I ended up using 5 features:
 
 FEATURE|SCORE
 --------|-------
 bonus|30.6522823057
 total_stock_value|10.8146348630
 shared_receipt_with_poi|10.6697373596
 exercised_stock_options|9.9561675821
 total_payments|8.9627155010
  
As we can see, most significant feature by far was size of bonus. Own feature did not make into the top 10 most significant features, it is 10th most significant feature according to SelectKBest() algorithm.

** 3) What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms? **

I tried four different algorithms: 

* Naive Bayes
* Decision tree 
* Support Vector Machines with linear kernel
* Random forest.

Performance criterion is F1 score. Using 10 best features (selected with SelectKBest function), PCA and MinMax scaler, I got following F1 scores:
 
Algorithm | F1 score
----------|----------
Naive Bayes | 0.50
Decision tree | 0.18
SVM, linear kernel | 0.25
Random forest | 0.44

Naive Bayes and Random Forest algorithms performed best, while SVM with linear kernel and Decision tree showed significantly lower F1 scores.
