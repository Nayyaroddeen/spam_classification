# spam_classification
This project contains the source files for the classification algorithms such as
Logistic Regression, Random Forest, Neural Network, Convolution Neural Network for
spam classification.

## Dataset Preparation
The dataset is downloaded from this (http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/index.html) location. In particular
Enron1 (http://nlp.cs.aueb.gr/software_and_datasets/Enron-Spam/preprocessed/enron1.tar.gz) from the Enron-Spam in pre-processed form.
While preparing the dataset I've selected  10% of the data as test set randomly and remaining 90% of the data  for the training and validation sets.

## Training and Validation file names are included in following file
    1.train_validation_sets.txt
## Test file names are included in following file
    1.test_set.txt

## Feature Representation Techniques
### I used following techniques to represent features.
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
Logistic Regression. It is observed that the validation accuracy is close to 97% in each fold and test accuracy is also around 96-97 %. Then I wanted to improve the
test accuracy so I moved to classifiers such as Random Forest, Neural Network, and Convolution Neural Network. I performed experiments
with each of these classifiers with different features. In all the experiments I found that Neural Network produced maximum test accuracy of 98.84%.

## Experimental Setup
### Feature extraction
Training model is build on the training data for feature extraction. The same model is used on test data to transform it into features. This above process is done for Word Count Feature Vector and Tf-Idf. Whereas for the
graph based feature extraction techniques we save the list of important words (Nodes). Then we create a One-Hot encoding for the each document using these important words.
### Model Selection
In each of the experiments five fold cross validation is used. Out of all the 5-fold cross validation we take the model that produced
highest test accuracy as the best model. The best model is saved and used on the unseen test data to check the performance.

### Model Testing
I used the saved models (best model out of 5-fold) to transform the test data into features and to test the classification algorithms.

## Results
As there are 4 algorithms and 4 feature extraction techniques, There are 16 experiments in total.
Among all these experiments only Neural Network with word count based features out performed by reaching 98.84% of testing accuracy.
Whereas all other classifiers could able to reach the accuracy with 90-97 % of accuracy. Tf-idf produced worst results (This part needs more tuning) among all the features representation techniques.

## Intresting Observations
    Graph coloring  based technique uses only 154 features and kcore based algorithms 198 featues but
    produces close to 95% accuracy. Time requried to build and test these models is much less comparing with
    other feature representaion techniques. Whereas wordcount and tfidf contains featues of length 40K+ features.

# Parameter Tuning
    1. For NN model I tuned the threshold from 0.5 to 0.20 found the best result at 0.25
    2. For Random Forest I tried increasing the number of trees as the number of trees increases the accuracy increased.
    3. For CNN tried with different filter sizes and layers but as the data is less I din't go further to explore much.


# How to Build Models
Please use Anaconda to install the following packages
## Required Packages
    1.pickle
    2.sklearn
    3.keras
    4.networkx
## Steps
    1.Once you clone the repo. run mkdir raw_input in spam_classification/ folder
    2.place the .ham.txt and .spam.txt files from the spam/ and ham/ folders of enron1 into raw_input.
    3.run python bulid_models.py to build model (This will build Neural Network Model) / python graph_based.py for the graph based model
    4.run python test_models.py to test the model

    Also note that code is commented for the rest of the experiments. As Neural network with word count features performed better
    I've considered it as a final algorithm. If you want to run some more additional experiments please un-comment respective the the classifier
    function to train and test the models.

## repo structure
    1) raw_input/ : This folder should contain .spam.txt and .ham.txt files
    2) best_models/: This folder is a destination folder for the models built in the 5-fold cross validation
    3) bulid_models.py : This file contains code for building the models
    4) graph_based.py : This file contains the code for building the models using the graph based features.
    5) test_models.py: This file contains the code for testing the models
    6) pre-process.py: This file splits the input file list to train, validation and test sets randomly.
    7) test_set.txt/train_validation_sets.txt: These are the outputs of the preprocess.py file.

