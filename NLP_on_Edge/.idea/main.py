import nltk
import keras
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras import backend as K
from keras import callbacks
from keras import layers
from keras import models

from keras.wrappers.scikit_learn import KerasClassifier
import pandas as pd
import tensorflow as tf
from sklearn import metrics
from sklearn import pipeline
from sklearn import preprocessing
from sklearn.externals import joblib
#nltk.download('treebank')
def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    words = sentence[index];
    return {
    'words': words,
    'is_first': index == 0,
    'is_last': index == len(sentence) - 1,
    'is_capitalized': words[0].upper() == words[0],
    'is_all_caps': words.upper() == words,
    'is_all_lower': words.lower() == words,
    'prefix-1': words[0],
    'prefix-2': words[:2],
    'prefix-3': words[:3],
    'prefix-4': words[:4],
    'suffix-1': words[-1],
    'suffix-2': words[-2:],
    'suffix-3': words[-3:],
    'prev_word': '' if index == 0 else sentence[index - 1],
    'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
    #'is_numeric': words.isdigit(),

}



tagged_sentences = nltk.corpus.treebank.tagged_sents()

train, test = sklearn.model_selection.train_test_split(tagged_sentences, test_size=0.2, random_state=1);
train, val = sklearn.model_selection.train_test_split(train, test_size=0.2, random_state=1);
def untag(tagged_sentence):
    return[w for w, t in tagged_sentence];
def transform(sentence):

    X, y = [],[];

    for x in sentence:

        for i in range(len(x)):
            X.append(features(untag(x),i));
            y.append(x[i][1]);


    return X, y;

X_train, y_train = transform(train);
X_test, y_test = transform(test);
X_val, y_val = transform(val);

vectorizer = DictVectorizer(sparse=False);
vectorizer.fit(X_train + X_test + X_val);

X_train =vectorizer.transform(X_train);
X_test =vectorizer.transform(X_test);
X_val =vectorizer.transform(X_val);

encoder = LabelEncoder();
encoder.fit(y_train + y_test + y_val);

y_train = encoder.transform(y_train);
y_test = encoder.transform(y_test);
y_val = encoder.transform(y_val);

from keras.utils import np_utils

y_train =np_utils.to_categorical(y_train);
y_test = np_utils.to_categorical(y_test);
y_val = np_utils.to_categorical(y_val);

#X contains the words

#start a tensorflow session
sess = tf.Session();
#use keras backend
K.set_session(sess);


def model(input_dim, hidden_neurons, output_dim):
    model = models.Sequential([
        layers.Dense(hidden_neurons, input_dim=input_dim, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(hidden_neurons, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(output_dim, activation='sigmoid')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

mod = model(X_train.shape[1], 512, y_train.shape[1])

mod.fit(X_train, y_train, epochs=5, validation_split =0.1, batch_size = 128)



score = mod.evaluate(X_test, y_test)
print("The score is: %f",score[1])

def translate(sent):
    l =[]
    index = 0
    for i in sent:
        l.append(features(sent,index))
        index += 1

    v = vectorizer.transform(l)

    encoded_tags = mod.predict_classes(v)

    return encoder.inverse_transform(encoded_tags)

translate(["He","is", "smart", "and", "likes","to", "climb"])