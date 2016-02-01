Enron Submission Free-Response Questions
==============


**1) Summarize the goal of this project and how machine learning is useful in trying to accomplish it.**

This project is a grading project for Udacity course "Introduction to Machine Learning". It deals with a dataset released in connection to famous Enron fraud scandal. Dataset contains information about key figures in Enron company at the time of that scandal, such as various compensation figures, data from email correspondence, etc. Goal of the project is to predict whether a person from dataset was convicted after fraud scandal or not using machine learning techniques.

**2) What features did you end up using in your POI identifier, and what selection process did you use to pick them?** 

In the beginning I used intuition to select features for my classifier. I started with 'total_payments' and 'from_poi_to_this_person', which produced reasonable accuracy but abysmal recall and precision. Then I decided to include all the features from the dataset. Additionally I added my own feature, share of bonus to total payments. 

Since features had vastly different scale, I impelmented a min-max scaler using sklearn functions, with default parameters.   

To select best features I used `SelectKBest()` function from sklearn package. 10 most significant features, with their scores are as follows:
 
 FEATURE|SCORE
 --------|-------
 bonus|30.6522823057
 total_stock_value|10.8146348630
 shared_receipt_with_poi|10.6697373596
 exercised_stock_options|9.9561675821
 total_payments|8.9627155010
 deferred_income|8.4934970305
 restricted_stock|8.0511018970
 long_term_incentive|7.5345222400
 loan_advances|7.0379327982
 bonus_share|5.7907234193
  
As we can see, most significant feature by far was size of bonus. Own feature is also made into the top 10 most significant features, thus it was used in my classifying algorithm.




