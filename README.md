## Detecting-Incongruent-Headlines-News-with-Auxiliary-Textual-Information

This is the implementation of Detecting Incongruent Headline News with Auxiliary Textual Information.

Auxiliary textual information contains subtitle and image caption.

## Pre-requisite

* tensorflow 2.4.0
* tensorflow-gpu 2.4.0
* numpy 1.19.4 
* pandas 1.1.5
* konlpy 0.5.2
* gensim 3.8.3
* nltk 3.5
* scikit-learn 0.24.0
* matplotlib 3.3.3

## DATASET

We only upload indexed test data due to memory issue.
We are also providing the dataset for non-commercial research purposes only.
Please request through ___

## Files description

* data : codes for data preprocessing, dataset generation, and input generation 
* src : files with model weights
* utils : codes for customize callbacks

## Instructions to run the project

* Training python3 main.py --mode train
* Testing python3 main.py --mode test
