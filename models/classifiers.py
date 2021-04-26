import numpy as np
from numpy.ma.core import set_fill_value
# You need to build your own model here instead of using well-built python packages such as sklearn

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import Perceptron
# You can use the models form sklearn packages to check the performance of your own models

class HateSpeechClassifier(object):
    """Base class for classifiers.
    """
    def __init__(self):
        pass
    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass
    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass


class AlwaysPreditZero(HateSpeechClassifier):
    """Always predict the 0
    """
    def predict(self, X):
        return [0]*len(X)

# TODO: Implement this
class NaiveBayesClassifier(HateSpeechClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        self.classes = {}
        self.trained = {}
        

    """Train your model based on training set
    
    Arguments:
        X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
    """
    def fit(self, X, Y):
        num_sentences = len(X)

        classes = np.unique(Y)
        class_counts = { 0: 0, 1: 0 }
        sentence_counts = { 0: 0, 1: 0 }
        fractions = { 0: np.ones(len(X[0])), 1: np.ones(len(X[0])) }
        for i in range(0, len(Y)):
            val = Y[i]
            feature_row =  X[i]
            sum = np.sum(feature_row)
            class_counts[val] += sum
            sentence_counts[val] += 1
            for f in range(0, len(feature_row)):
                fractions[val][f] += feature_row[f]
        total_count = len(X[0])
        denominators = { 0: np.full(len(X[0]), class_counts[0] + total_count), 1: np.full(len(X[0]), class_counts[1] + total_count) }
            

        resulting_values = { 0: (fractions[0] / denominators[0]), 1: (fractions[1] / denominators[1]) }
        self.classes = sentence_counts
        self.trained = resulting_values
        
    
    def predict(self, X):
        to_ret = []
        dict = {}
        for c in range(0, len(self.classes)):
            dict[c] = []
            for i in range(0, len(X)):
                total = (self.classes[c] / len(X))
                feature_row = X[i]
                a = np.nonzero(feature_row)
                total *= np.prod(np.multiply(feature_row[a], self.trained[c][a]))
                dict[c].append(np.log(total))
        for i in range(0, len(dict[0])):
            if (dict[0][i] > dict[1][i]):
                to_ret.append(0)
            else:
                to_ret.append(1)
        return to_ret
        
    
    def getImportantTen(self, unigram):
        ratios = (self.trained[1] / self.trained[0])
        top_ten = np.empty(10, dtype=np.dtype('U100'))
        bottom_ten = np.empty(10, dtype=np.dtype('U100')) 

        x = ratios.argsort()[-10:][::-1]
        y = ratios.argsort()[:10]
        top_ten_ratio = np.zeros(10)
        bottom_ten_ratio = np.zeros(10)
        for i in range(0, 10):
            word = self.__getIndexFromValue(unigram, x[i])
            top_ten[i] = word
            top_ten_ratio[i] = ratios[x[i]]
            word = self.__getIndexFromValue(unigram, y[i])
            bottom_ten[i] = word
            bottom_ten_ratio[i] = ratios[y[i]]
        
        print("")
        print("===== Top Ten Ratios =====")
        print("-------------------------------------------")
        print('{0:25}  {1}'.format('   Word', 'Ratio'))
        print("-------------------------------------------")
        for x in range(1, 11):
            print('{0:20}  {1}'.format(str(x) + ") " + top_ten[x - 1], str(top_ten_ratio[x-1])))

        print("")
        print("===== Bottom Ten Ratios =====")
        print("-------------------------------------------")
        print('{0:25}  {1}'.format('   Word', 'Ratio'))
        print("-------------------------------------------")
        for x in range(1, 11):
            print('{0:20}  {1}'.format(str(x) + ") " + bottom_ten[x - 1], str(bottom_ten_ratio[x-1])))
        print("")

    def __getIndexFromValue(self, unigram, val):
        for k, v in unigram.items():
         if (val == v):
             return k


# TODO: Implement this
class LogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """
    def __init__(self):
        self.alpha = .75
        self.weights = []
        self.b = .2
        self.iterations = 1200
        self.l = .1

    def __calc_prob(self, x, w, b):
        weighted_sum = np.sum(np.multiply(w, x)) + b
        sigmoid = (1 / (1 + np.exp(-1 * weighted_sum)))
        return sigmoid

    def __modify_weights(self, x, a, y):
        first = np.full(len(x), a)
        first = first - y
        res = np.multiply(x, first)
        return res
    
    def fit(self, X, Y):
        weights = np.full(len(X[0]), (1 / len(X[0])))
        updated_weights = np.zeros(len(X[0]))

        for i in range(0, self.iterations):
            for x in range(0, len(X)):
                calc_prob = self.__calc_prob(X[x], weights, self.b)
                new_weight = self.__modify_weights(X[x], calc_prob, Y[x])
                updated_weights = np.add(updated_weights, new_weight)
            updated_weights /= len(X)
            updated_weights = self.alpha * updated_weights
            updated_weights = updated_weights - (self.l * np.square(updated_weights)) #L2 regularization
            weights = weights - updated_weights

        self.weights = weights

    def predict(self, X):
        to_ret = []
        for x in range(0, len(X)):
            calc = self.__calc_prob(X[x], self.weights, self.b)
            if (calc > 0.5):
                to_ret.append(1)
            else:
                to_ret.append(0)
        
        return to_ret
