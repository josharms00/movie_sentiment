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



class AlgoVote(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []

        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        return mode(votes)


def preprocessing():
    # create lammentizer
    lemmatizer = WordNetLemmatizer()

    # create set of useless words
    stop_words = set(stopwords.words("english")) 

    # import training data
    # 5300 of each review
    pos_rev = open("movie review data/positive.txt", "r").read()
    neg_rev = open("movie review data/negative.txt", "r").read()

    documents = []

    for r in pos_rev.split("\n"):
        documents.append( (r, "pos") )

    for r in neg_rev.split("\n"):
        documents.append( (r, "neg") )

    all_words = []

    for d in documents:
        _words = word_tokenize(d[0])
        for w in _words:
            if w not in stop_words:
                all_words.append(lemmatizer.lemmatize(w.lower()))

    # convert to frequncy distribution of words
    all_words = nltk.FreqDist(all_words)



if __name__ == "__main__":
    