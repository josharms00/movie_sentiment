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
from sklearn.linear_model import SGDClassifier
from statistics import mode
import os


word_features = []
documents = []



# returns frequency distribution
def preprocessing(documents):
    # create lammentizer
    lemmatizer = WordNetLemmatizer()

    # create set of useless words
    stop_words = set(stopwords.words("english")) 

    adjectives = ["JJ", "JJR", "JJS"]

    all_words = []

    for d in documents:
        _words = word_tokenize(d[0])
        pos = nltk.pos_tag(_words)
        for w in pos:
            if w not in stop_words and w[1] in adjectives:
                all_words.append(lemmatizer.lemmatize(w[0].lower()))

    # convert to frequncy distribution of words
    all_words = nltk.FreqDist(all_words)

    return all_words

if os.path.isfile("movie review data/word_features.pickle"):
    _feats = open("movie review data/word_features.pickle", "rb")
    word_features = pickle.load(_feats)
    _feats.close()
else:
    # import training data
    # 5300 of each review
    pos_rev = open("movie review data/positive.txt", "r").read()
    neg_rev = open("movie review data/negative.txt", "r").read()

    for r in pos_rev.split("\n"):
        documents.append( (r, "pos") )

    for r in neg_rev.split("\n"):
        documents.append( (r, "neg") )

    all_words = preprocessing(documents)

    # convert to list of 5000 most common words
    word_features = list(all_words.keys())[:5000]

    save_feats = open("movie review data/word_features.pickle", "wb")
    pickle.dump(word_features, save_feats)
    

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

# determines what words are important to classification
def find_features(document):
    words = set(word_tokenize(document))
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

def train_and_pickle(docs):
    random.shuffle(docs)

    # convert the document into feature list to see what words show up more often in bad review vs good reviews
    featureset = [ (find_features(doc), cat) for (doc, cat) in docs]

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

    LinearSVC_classifier.train(train_set)

    save_classifier = open("trained_classifiers/linearsvc.pickle", "wb")
    pickle.dump(LinearSVC_classifier, save_classifier)

    print("LinearSVC Accurarcy: ", (nltk.classify.accuracy(LinearSVC_classifier, test_set)))

    # stochastic gradient descent
    SGDClassifier_classifier = SklearnClassifier(SGDClassifier())

    SGDClassifier_classifier.train(train_set)

    save_classifier = open("trained_classifiers/sgd_classifier.pickle", "wb")
    pickle.dump(SGDClassifier_classifier, save_classifier)

    print("SGDClassifier Accurarcy: ", (nltk.classify.accuracy(SGDClassifier_classifier, test_set)))

# load all pickled models and create class object
if os.path.isfile("trained_classifiers/naivebayes.pickle"):

    open_classifier = open("trained_classifiers/naivebayes.pickle", "rb")
    NaiveBayes_classifier = pickle.load(open_classifier)
    open_classifier.close()

    open_classifier = open("trained_classifiers/mn_naivebayes.pickle", "rb")
    MultinomialNB_classifier = pickle.load(open_classifier)

    open_classifier = open("trained_classifiers/bern_naivebayes.pickle", "rb")
    BernoulliNB_classifier = pickle.load(open_classifier)
    open_classifier.close()

    open_classifier = open("trained_classifiers/linearsvc.pickle", "rb")
    LinearSVC_classifier = pickle.load(open_classifier)
    open_classifier.close()

    open_classifier = open("trained_classifiers/sgd_classifier.pickle", "rb")
    SGDClassifier_classifier = pickle.load(open_classifier)
    open_classifier.close()

    voter = AlgoVote(NaiveBayes_classifier, MultinomialNB_classifier, BernoulliNB_classifier,
                    LinearSVC_classifier, SGDClassifier_classifier)
else:
    print("Algorithms have not been trained")

# analze outside data
def sentiment(text):
    feats = find_features(text)

    return voter.classify(feats)


# docs = documents

# train_and_pickle(docs)

# f = find_features("This movie was awesome! The acting was conan!")

# print(SGDClassifier_classifier.classify(f))

    

   

    

    