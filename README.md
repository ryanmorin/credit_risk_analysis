# Credit Risk Analysis

## Overview
The project uses a dataset from LendingClub to train six machine learning models to predict the risk of a particular loan. The models are coming from two python libraries: imbalanced-learn and scikit-learn.

## Results
### Oversampling

#### Naive Random Oversampling (NRO)
Random oversampling is one way to deal with biased data. Random oversampling randomly duplicates examples of the minority class in the training set.

* Balanced Accuracy Score (Precision + Recall / 2) = 0.63
* Precision (TP / (TP + FP)) = 0.0091 - The model recognizes a large number of false positives
* Recall (TP / (TP + FN)) = 0.6 - The model does a better job identifying True positives.

The model identified a large number of false positives which hurt the model's precision.

![NRO](https://github.com/ryanmorin/credit_risk_analysis/blob/main/NRO.png)

#### SMOTE Oversampling (SMOTE)
In SMOTE, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created.

* Balanced Accuracy Score (Precision + Recall / 2) = 0.64
* Precision (TP / (TP + FP)) = 0.0089 - The model recognizes a large number of false positives
* Recall (TP / (TP + FN)) = 0.66 - The model does a better job identifying True positives.

The model identified a large number of false positives which hurt the model's precision.

![SMOTE](https://github.com/ryanmorin/credit_risk_analysis/blob/main/SMOTE.png)

### Undersampling
#### Cluster Centroids
This technique undersamples the dataset by reducing the number of major class data points by substituing the cluster for one data point the algorithm estimates is at the center of the cluster.

* Balanced Accuracy Score (Precision + Recall / 2) = 0.64
* Precision (TP / (TP + FP)) = 0.005 - The model recognizes a large number of false positives
* Recall (TP / (TP + FN)) = 0.61 - The model does a better job identifying True positives.

The model identified a large number of false positives which hurt the model's precision.

![CC](https://github.com/ryanmorin/credit_risk_analysis/blob/main/CC.png)

### Combo Over & Under Sampling
#### SMOTEENN
Oversampling methods duplicate or create new synthetic examples in the minority class, whereas undersampling methods delete or merge examples in the majority class. Both types of resampling can be effective when used in isolation, although can be more effective when both types of methods are used together.

* Balanced Accuracy Score (Precision + Recall / 2) = 0.65
* Precision (TP / (TP + FP)) = 0.0084 - The model recognizes a large number of false positives
* Recall (TP / (TP + FN)) = 0.75 - The model does a better job identifying True positives.

The model identified a large number of false positives which hurt the model's precision.

![SMOTEENN](https://github.com/ryanmorin/credit_risk_analysis/blob/main/SMOTEENN.png)

### Ensemble Models
#### Random Forest Classifiers (RF)
The key principle underlying the random forest approach comprises the construction of many “simple” decision trees in the training stage and the majority vote (mode) across them in the classification stage.

* Balanced Accuracy Score (Precision + Recall / 2) = 0.80
* Precision (TP / (TP + FP)) = 0.04 - The model recognizes a large number of false positives
* Recall (TP / (TP + FN)) = 0.69 - The model does a better job identifying True positives.

The model identified a large number of false positives which hurt the model's precision.

![RF](https://github.com/ryanmorin/credit_risk_analysis/blob/main/RF.png)

#### Easy Ensemble Classifier (EE)
The Easy Ensemble involves creating balanced samples of the training dataset by selecting all examples from the minority class and a subset from the majority class.

* Balanced Accuracy Score (Precision + Recall / 2) = 0.93
* Precision (TP / (TP + FP)) = 0.08 - The model recognizes a large number of false positives
* Recall (TP / (TP + FN)) = 0.91 - The model does a better job identifying True positives.

The model identified a large number of false positives which hurt the model's precision.

![EEC](https://github.com/ryanmorin/credit_risk_analysis/blob/main/EE.png)

## Conclusion
All models have a very low level of precision when it comes to predicting high-risk loans. The EE model has the best recall number and Balance Accuracy Score. Yet the EE like the other models has a difficult time predicting a true high-risk loan.  Many loans (over 90%) that it predicts as high-risk look like a high-risk loan but never become one. Therefore none of these models should be used.
