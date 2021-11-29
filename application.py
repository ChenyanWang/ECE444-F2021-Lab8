# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 15:56:53 2019

@author: charl
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import flask
from flask import Flask, Response, Request, json

#Cleaning libraries
from bs4 import BeautifulSoup
import re
import itertools
import emoji

##############################################################################################
#
#   INIT
#
##############################################################################################

##############################################################################################
#
#   CLEANING FUNCTIONS
#
##############################################################################################


def strip_accents(text):
    #length_initial=len(text)
    #initial_text=text
    if 'ø' in text or  'Ø' in text:
        return text   
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

def load_dict_smileys():
    
    return {
        ":‑)":"smiley",
        ":-]":"smiley",
        ":-3":"smiley",
        ":->":"smiley",
        "8-)":"smiley",
        ":-}":"smiley",
        ":)":"smiley",
        ":]":"smiley",
        ":3":"smiley",
        ":>":"smiley",
        "8)":"smiley",
        ":}":"smiley",
        ":o)":"smiley",
        ":c)":"smiley",
        ":^)":"smiley",
        "=]":"smiley",
        "=)":"smiley",
        ":-))":"smiley",
        ":‑D":"smiley",
        "8‑D":"smiley",
        "x‑D":"smiley",
        "X‑D":"smiley",
        ":D":"smiley",
        "8D":"smiley",
        "xD":"smiley",
        "XD":"smiley",
        ":‑(":"sad",
        ":‑c":"sad",
        ":‑<":"sad",
        ":‑[":"sad",
        ":(":"sad",
        ":c":"sad",
        ":<":"sad",
        ":[":"sad",
        ":-||":"sad",
        ">:[":"sad",
        ":{":"sad",
        ":@":"sad",
        ">:(":"sad",
        ":'‑(":"sad",
        ":'(":"sad",
        ":‑P":"playful",
        "X‑P":"playful",
        "x‑p":"playful",
        ":‑p":"playful",
        ":‑Þ":"playful",
        ":‑þ":"playful",
        ":‑b":"playful",
        ":P":"playful",
        "XP":"playful",
        "xp":"playful",
        ":p":"playful",
        ":Þ":"playful",
        ":þ":"playful",
        ":b":"playful",
        "<3":"love"
        }


def load_dict_contractions():
    
    return {
        "ain't":"is not",
        "amn't":"am not",
        "aren't":"are not",
        "can't":"cannot",
        "'cause":"because",
        "couldn't":"could not",
        "couldn't've":"could not have",
        "could've":"could have",
        "daren't":"dare not",
        "daresn't":"dare not",
        "dasn't":"dare not",
        "didn't":"did not",
        "doesn't":"does not",
        "don't":"do not",
        "e'er":"ever",
        "em":"them",
        "everyone's":"everyone is",
        "finna":"fixing to",
        "gimme":"give me",
        "gonna":"going to",
        "gon't":"go not",
        "gotta":"got to",
        "hadn't":"had not",
        "hasn't":"has not",
        "haven't":"have not",
        "he'd":"he would",
        "he'll":"he will",
        "he's":"he is",
        "he've":"he have",
        "how'd":"how would",
        "how'll":"how will",
        "how're":"how are",
        "how's":"how is",
        "I'd":"I would",
        "I'll":"I will",
        "I'm":"I am",
        "I'm'a":"I am about to",
        "I'm'o":"I am going to",
        "isn't":"is not",
        "it'd":"it would",
        "it'll":"it will",
        "it's":"it is",
        "I've":"I have",
        "kinda":"kind of",
        "let's":"let us",
        "mayn't":"may not",
        "may've":"may have",
        "mightn't":"might not",
        "might've":"might have",
        "mustn't":"must not",
        "mustn't've":"must not have",
        "must've":"must have",
        "needn't":"need not",
        "ne'er":"never",
        "o'":"of",
        "o'er":"over",
        "ol'":"old",
        "oughtn't":"ought not",
        "shalln't":"shall not",
        "shan't":"shall not",
        "she'd":"she would",
        "she'll":"she will",
        "she's":"she is",
        "shouldn't":"should not",
        "shouldn't've":"should not have",
        "should've":"should have",
        "somebody's":"somebody is",
        "someone's":"someone is",
        "something's":"something is",
        "that'd":"that would",
        "that'll":"that will",
        "that're":"that are",
        "that's":"that is",
        "there'd":"there would",
        "there'll":"there will",
        "there're":"there are",
        "there's":"there is",
        "these're":"these are",
        "they'd":"they would",
        "they'll":"they will",
        "they're":"they are",
        "they've":"they have",
        "this's":"this is",
        "those're":"those are",
        "'tis":"it is",
        "'twas":"it was",
        "wanna":"want to",
        "wasn't":"was not",
        "we'd":"we would",
        "we'd've":"we would have",
        "we'll":"we will",
        "we're":"we are",
        "weren't":"were not",
        "we've":"we have",
        "what'd":"what did",
        "what'll":"what will",
        "what're":"what are",
        "what's":"what is",
        "what've":"what have",
        "when's":"when is",
        "where'd":"where did",
        "where're":"where are",
        "where's":"where is",
        "where've":"where have",
        "which's":"which is",
        "who'd":"who would",
        "who'd've":"who would have",
        "who'll":"who will",
        "who're":"who are",
        "who's":"who is",
        "who've":"who have",
        "why'd":"why did",
        "why're":"why are",
        "why's":"why is",
        "won't":"will not",
        "wouldn't":"would not",
        "would've":"would have",
        "y'all":"you all",
        "you'd":"you would",
        "you'll":"you will",
        "you're":"you are",
        "you've":"you have",
        "Whatcha":"What are you",
        "luv":"love"
        }

def tweet_cleaning_for_sentiment_analysis(tweet):    
    #Escaping HTML characters
    tweet = BeautifulSoup(tweet).get_text()
    tweet = tweet.replace('\x92',"'")
    
    #REMOVAL of hastags/account
    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", tweet).split())
    #Removal of address
    tweet = ' '.join(re.sub("(\w+:\/\/\S+)", " ", tweet).split())
    #Removal of Punctuation
    tweet = ' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ", tweet).split())
    ## LOWER CASE
    tweet = tweet.lower()
    
    #Apostrophe Lookup #https://en.wikipedia.org/wiki/Contraction_%28grammar%29
    APPOSTOPHES = load_dict_contractions()
    tweet = tweet.replace("’","'")
    words = tweet.split()
    reformed = [APPOSTOPHES[word] if word in APPOSTOPHES else word for word in words]
    tweet = " ".join(reformed)
    tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))
    
    
    #Deal with EMOTICONS
    #https://en.wikipedia.org/wiki/List_of_emoticons
    SMILEY = load_dict_smileys() #{"<3" : "love", ":-)" : "smiley", "" : "he is"}
    words = tweet.split()
    reformed = [SMILEY[word] if word in SMILEY else word for word in words]
    tweet = " ".join(reformed)
    tweet = emoji.demojize(tweet)
    
    #Strip accents  
    tweet= strip_accents(tweet)
    tweet = tweet.replace(":"," ")
    tweet = ' '.join(tweet.split())    
    return tweet



##############################################################################################
#
#   INFERENCE SYSTEM
#
##############################################################################################

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    loaded_model = None
    vectorizer = None
    

    @classmethod
    def get_models(cls):
        if cls.loaded_model == None:
            with open('basic_classifier.pkl', 'rb') as fid:
                cls.loaded_model = pickle.load(fid)

        if cls.vectorizer == None:
            with open('count_vectorizer.pkl', 'rb') as vd:
                cls.vectorizer = pickle.load(vd)   
                
        return (cls.loaded_model, cls.vectorizer)


    @classmethod
    def predict(cls, text_input):
        """For the input, do the predictions and return them.
        Args:
           Dictionary with text to analyze and language in which it is written."""
        loaded_model, vectorizer = cls.get_models()
        prediction = loaded_model.predict(vectorizer.transform([text_input]))[0]
        return prediction



##############################################################################################
#
#   FLASK APP
#
##############################################################################################

# The flask app for serving predictions
application = app = Flask(__name__)


@application.route('/')
def hello():
    return "Welcome to your own Sentiment Analysis Tool"

@application.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_models() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@application.route('/invocations', methods=['GET'])
def transformation():
    """Do an inference on a single batch of data.
    """
    data = None

    if flask.request.content_type == 'application/json':
        #data = request.data.decode('utf-8')
        data= flask.request.get_json()

    else:
        return flask.Response(response='This predictor only supports Json data', status=415, mimetype='text/plain')

    text = tweet_cleaning_for_sentiment_analysis(data.get('text'))

    # Do the prediction
    final_sentiment = ScoringService.predict(text)
    
    result = {'Sentiment': final_sentiment}    
    
    return flask.Response(response=json.dumps(result), status=200, mimetype='application/json')

if __name__ == '__main__':
    application.debug = True
    application.run()