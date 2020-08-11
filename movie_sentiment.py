import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.classify import ClassifierI
from statistics import mode
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import 
from statistics import mode

# create lammentizer
lemmatizer = WordNetLemmatizer()

class AlgoVote(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []

        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        return mode(votes)

# import training data
# 5300 of each review
pos_rev = open("movie review data/positive.txt", "r").read()
neg_rev = open("movie review data/negative.txt", "r").read()