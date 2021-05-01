import pandas as pd
from classifiers import *
from utils import *
import numpy as np
import time
import argparse

#python main.py --model LogisticRegression

# provided to me
def accuracy(pred, labels):
    correct = (np.array(pred) == np.array(labels)).sum()
    accuracy = correct/len(pred)
    print("Accuracy: %i / %i = %.4f " %(correct, len(pred), correct/len(pred)))

# provided to me
def read_data(path):
    train_frame = pd.read_csv(path + 'kory_data.csv')

    # You can form your test set from train set
    # We will use our test set to evaluate your model
    try:
        test_frame = pd.read_csv(path + 'kory_test.csv')
    except:
        test_frame = train_frame

    return train_frame, test_frame


# written by Kory Brantley
def calc_counts(ticker, predictions, frame):
    total = 0
    count = 0
    for x in range(0, len(predictions)):
        pred = predictions[x]
        sentence = frame[x]
        if ticker in sentence:
            total = total + 1
            if pred == 1:
                count = count + 1

    return count, total
        

# provided to me
def main():

    flag = True
    while(flag):
        ticker = input("Enter stock ticker: ")
        if (type(ticker) == str):
            flag = False

    print("Starting classification:")
    train_frame, test_frame = read_data('./data/')

    feat_extractor = UnigramFeature()
    # Tokenize text into tokens
    tokenized_text = []
    for i in range(0, len(train_frame['text'])):
        tokenized_text.append(tokenize(str(train_frame['text'][i])))

    feat_extractor.fit(tokenized_text)

    # form train set for training
    X_train = feat_extractor.transform_list(tokenized_text)
    Y_train = train_frame['label']


    # form test set for evaluation
    tokenized_text = []
    for i in range(0, len(test_frame['text'])):
        tokenized_text.append(tokenize(test_frame['text'][i]))
    X_test = feat_extractor.transform_list(tokenized_text)
    Y_test = test_frame['label']


    model = LogisticRegressionClassifier()

    start_time = time.time()
    model.fit(X_train,Y_train)
    print("===== Train Accuracy =====")
    prediction = model.predict(X_train)
    accuracy(prediction, Y_train)
    
    print("===== Test Accuracy =====")
    prediction = model.predict(X_test)
    accuracy(prediction, Y_test)

    # written by Kory Brantley
    c, t = calc_counts(ticker, prediction, test_frame['text'])

    print("\n===== Accuracy for " + str(ticker) + " =====")
    print("Accuracy: %i / %i = %.4f " % (c, t, c/t))

    print("\nTime for training and test: %.2f seconds" % (time.time() - start_time))


# provided to me
if __name__ == '__main__':
    main()