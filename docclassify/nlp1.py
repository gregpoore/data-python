#!/usr/bin/python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
#from nltk.stem.snowball import SnowballStemmer
#from nltk.stem.snowball import EnglishStemmer
 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB


# Stem all words
def parseTextString(stemmer, text_string):
    words = [stemmer.stem(word) for word in text_string.split()]
    words = " ".join(words)
    return words


def main():
    #stemmer = SnowballStemmer('english')
    #stemmer = EnglishStemmer()

    training_data=open('trainingdata.txt', 'rU')
    n = int(training_data.readline().strip())    
    
    train_data = []
    class_data = []

    for i in range(n):
        line = training_data.readline().strip()
        train_data.append(line[1:].strip())
        class_data.append(int(line[0]))
        
    train_data = np.array(train_data)
    class_data = np.array(class_data)


    # 2) Vectorize bag of words
    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.5, sublinear_tf=True )
    vectorizer.fit(train_data)
    X_train = vectorizer.transform(train_data)
        
  
    
    # Read test data from input
    X_test = np.array([raw_input().strip() for i in range(int(raw_input().strip()))])

    X_test = vectorizer.transform(X_test)

    clf = PassiveAggressiveClassifier(n_iter=9) 
    
    clf.fit(X_train, class_data)
    
    pred = clf.predict(X_test)
    for i in pred:
        print i
        

if __name__ == "__main__":
    main()

