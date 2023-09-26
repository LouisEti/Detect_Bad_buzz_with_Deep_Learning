# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:35:26 2022

@author: Loulou
"""

from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np 
import joblib
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import pickle


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import gensim
import gensim.models.keyedvectors as word2vec
from gensim.parsing.preprocessing import remove_stopwords
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models.coherencemodel import CoherenceModel
from gensim.parsing.preprocessing import remove_stopwords, strip_non_alphanum, strip_short


import keras 
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, Dropout, LSTM, Dense
from keras.optimizers import Adam
from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model


app = Flask(__name__)

# Load model and tokenizer
model = load_model('model_glove')
with open('tokenizer_glove.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

#Define all text_preprocessing functions 

negative_stopwords = ['no', 'don', "don't", 'nor', 'not', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', 
                      "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', 
                      "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
                      'wouldn', "wouldn't", "can't", 't'
                     ]

stopwords = set(stopwords.words('english')) - set(negative_stopwords)

def text_processing(serie):
    
    tokenizer = RegexpTokenizer(r'\w+')
    serie = serie.apply(lambda x: x.lower())
    serie = serie.apply(lambda x: tokenizer.tokenize(x))
    serie = serie.apply(lambda x: [word for word in x if word not in stopwords])
    serie = serie.apply(lambda x: ' '.join([word for word in x]))
    serie = serie.apply(strip_non_alphanum) #remove non alpha numeric character
    serie = serie.apply(lambda x: strip_short(x, minsize=3))
    
    return pd.DataFrame(serie)



# Load route
@app.route('/')
def index():
    return 'Hello'

@app.route('/predict',methods=['POST'])
def predict():
    new_review = [str(x) for x in request.form.values()]
    new_review = text_processing(pd.Series(new_review))
    
    new_review = tokenizer.texts_to_sequences(new_review.loc[0])
    new_review = pad_sequences(new_review, value=0, padding='post', maxlen=40)
    
    prediction = model.predict(new_review)
    
    if prediction > 0.5:
        # return(jsonify('Positive'))
        
        return jsonify(
            {
            'Score' : str(prediction[0][0]),
            'Nature commentaire' : 'Positive'
            })
        
    else:
        return jsonify(
            {'Score': str(prediction[0][0]),
             'Nature commentaire': 'Negative'
             }
            )

@app.route('/test',methods=['POST'])
def test():
    return [str(x) for x in request.form.values()]
    
    
app.run(host='127.0.0.1', port=5000)
