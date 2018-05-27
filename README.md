# spam_classification
This project contains the source files for the classification algorithms such as
Logistic Regression, Random Forest, Neural Network, Convolution Neural Network for
spam classification.

## Dataset Preparation
The dataset is downloaded from this(http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html) location. In particular
Enron1(http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/preprocessed/enron1.tar.gz) from the Enron-Spam in pre-processed form.
While preparing the dataset I've selected  10% of the data as test set randomly and remaining 90% of the for the training and validation sets.

## Feature Representation Techniques
### I used following techniques to represent the techniques.
    1. Word Count Feature Vector
    2. Kcore features
    3. Novel Graph Coloring based feature extraction
    4. Tf-Idf


## Classification Algorithms
### I used following classification algorithms
    1. Logistic Regression
    2. Random Forest
    3. Neural Network
    4. Convolution Neural Network

As the spam classification problem is a binary classification problem, I started with
Logistic Regression. It is observed that the test accuracy is close to 97% in each fold. Then I wanted to improve the
test accuracy so I moved to clasifiers such as Random Forest, Neural Network, Convolution Neural Network. I performed experiments
with each of these classifiers with different features. In all the experiments I found that Neural Network produced maximum test accuracy of 98%.

## Experimental Setup
### Feature extraction
Training model is build on the training data. The same model is used on test to create features. This above process is done for Word Count Feature Vector and Tf-Idf. Whereas for the
graph based feature extraction techniques we builds lists for the important words. Then we create a One-Hot encoding for the each document.
### Model Selection
In each of the experiments five fold cross validation is used. Out of all the 5-fold cross validation we take the model that produced
highest test accuracy as the best model. The best model is saved and used on the unseen test data to check the performance.
### Model Testing
I used the saved models to create features and test the classification algorithms.

## Results
As there are 4 algorithms and 4 feature extraction techniques, There are 16 experiments in total.
Among All these experiments only Neural Network with word count based features out performed by reaching 98% of testing accuracy.
Whereas all other classifiers could able to reach the accuracy with 90-97 % of accuracy.

## Intresting Observations
