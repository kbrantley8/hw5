import numpy as np
from numpy.ma.core import set_fill_value
# You need to build your own model here instead of using well-built python packages such as sklearn

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import Perceptron
# You can use the models form sklearn packages to check the performance of your own models

# provided to me
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

# written by Kory Brantley
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
