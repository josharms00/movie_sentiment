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

# classifies data based on the majority of the algorithms' choice
class AlgoVote(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []

        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        return mode(votes)

# returns frequency distribution
def preprocessing(documents):
    # create lammentizer
    lemmatizer = WordNetLemmatizer()

    # create set of useless words
    stop_words = set(stopwords.words("english")) 

    all_words = []

    for d in documents:
        _words = word_tokenize(d[0])
        for w in _words:
            if w not in stop_words:
                all_words.append(lemmatizer.lemmatize(w.lower()))

    # convert to frequncy distribution of words
    all_words = nltk.FreqDist(all_words)

    return all_words

# determines what words are important to classification
def find_features(document):
    words = set(word_tokenize(document))
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


def train_and_pickle(documents):

    random.shuffle(documents)

    # convert the document into feature list to see what words show up more often in bad review vs good reviews
    featureset = [ (find_features(doc), cat) for (doc, cat) in documents]

    train_set = featureset[:10000]
    test_set = featureset[10000:]

    # normal naive bayes
    NaiveBayes_classifier = nltk.NaiveBayesClassifier.train(train_set)

    save_classifier = open("trained_classifiers/naivebayes.pickle", "wb")
    pickle.dump(NaiveBayes_classifier, save_classifier)

    print("Naive Bayes Accurarcy: ", (nltk.classify.accuracy(NaiveBayes_classifier, test_set)))

    # multinomial naive bayes
    MultinomialNB_classifier = SklearnClassifier(MultinomialNB())

    MultinomialNB_classifier.train(train_set)

    save_classifier = open("trained_classifiers/mn_naivebayes.pickle", "wb")
    pickle.dump(MultinomialNB_classifier, save_classifier)

    print("Multinomial Naive Bayes Accurarcy: ", (nltk.classify.accuracy(MultinomialNB_classifier, test_set)))

    # bernoulli naive bayes
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())

    BernoulliNB_classifier.train(train_set)

    save_classifier = open("trained_classifiers/bern_naivebayes.pickle", "wb")
    pickle.dump(BernoulliNB_classifier, save_classifier)

    print("Bernoulli Naive Bayes Accurarcy: ", (nltk.classify.accuracy(BernoulliNB_classifier, test_set)))

    # linear state vector classifier
    LinearSVC_classifier = SklearnClassifier(LinearSVC())

    save_classifier = open("trained_classifiers/linearsvc.pickle", "wb")
    pickle.dump(LinearSVC_classifier, save_classifier)

    LinearSVC_classifier.train(train_set)
    print("LinearSVC Accurarcy: ", (nltk.classify.accuracy(LinearSVC_classifier, test_set)))

    # stochastic gradient descent
    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())

    save_classifier = open("trained_classifiers/sgd_classifier.pickle", "wb")
    pickle.dump(SGDClassifier_classifier, save_classifier)

    SGDClassifier_classifier.train(train_set)
    print("SGDClassifier Accurarcy: ", (nltk.classify.accuracy(SGDClassifier_classifier, test_set)))


if __name__ == "__main__":

    # import training data
    # 5300 of each review
    pos_rev = open("movie review data/positive.txt", "r").read()
    neg_rev = open("movie review data/negative.txt", "r").read()

    documents = []

    for r in pos_rev.split("\n"):
        documents.append( (r, "pos") )

    for r in neg_rev.split("\n"):
        documents.append( (r, "neg") )

    all_words = preprocessing(documents)

    # convert to list of 5000 most common words
    word_features = list(all_words.keys())[:5000]

    train_and_pickle(documents)

   

    

    