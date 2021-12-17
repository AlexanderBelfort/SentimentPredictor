from os import X_OK
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
# NLTK is very useful for natural language applications
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
import contractions
import afinn
# We use spacy for extracting useful information from English words
import spacy
nlp = spacy.load('en', parse = False, tag=False, entity=False)

# This dictionary will be used to expand contractions (e.g. we'll -> we will)
from contractions import contractions_dict
import re

# Unicodedata will be used to remove accented characters
import unicodedata

# BeautifulSoup will be used to remove html tags
from bs4 import BeautifulSoup

# Lexicon models
from afinn import Afinn
from nltk.corpus import sentiwordnet as swn
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Evaluation libraries
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix



from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)
model = pickle.load(open("lr_model.pkl", 'rb'))
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

        
    def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                        accented_char_removal=True, text_lower_case=True, 
                        text_lemmatization=True, special_char_removal=True, 
                        stopword_removal=True):



        normalized_corpus = []
        # normalize each document in the corpus
        for doc in corpus:
            # strip HTML
            if html_stripping:
                doc = strip_html_tags(doc)
            # remove accented characters
            if accented_char_removal:
                doc = remove_accented_chars(doc)
            # expand contractions    
            if contraction_expansion:
                doc = expand_contractions(doc)
            # lowercase the text    
            if text_lower_case:
                doc = doc.lower()
            # remove extra newlines
            doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
            # insert spaces between special characters to isolate them    
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            # lemmatize text
            if text_lemmatization:
                doc = lemmatize_text(doc)
            # remove special characters    
            if special_char_removal:
                doc = remove_special_characters(doc)  
            # remove extra whitespace
            doc = re.sub(' +', ' ', doc)
            # remove stopwords
            if stopword_removal:
                doc = remove_stopwords(doc, is_lower_case=text_lower_case)
                
            normalized_corpus.append(doc)
        return normalized_corpus


    def remove_stopwords(text, is_lower_case=False):
        tokenizer = ToktokTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        
        nltk.download('stopwords') # ignore the, a, an, etc
        stopword_list = nltk.corpus.stopwords.words('english')
        stopword_list.remove('no')
        stopword_list.remove('not')
        

        if is_lower_case:
            filtered_tokens = [token for token in tokens if token not in stopword_list]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)    
        return filtered_text



    # we see <br> so html tags, strip

    def strip_html_tags(text):
        soup = BeautifulSoup(text, "html.parser")
        stripped_text = soup.get_text()
        return stripped_text

    def remove_accented_chars(text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return text

    def remove_special_characters(text):
        text = re.sub('[^a-zA-z0-9\s]', '', text)
        return text

    def expand_contractions(text, contraction_mapping=contractions_dict):
        
        contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())), 
                                        flags=re.IGNORECASE|re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(match) else contraction_mapping.get(match.lower())                               
            return first_char+expanded_contraction[1:] if expanded_contraction != None else match
            
        expanded_text = contractions_pattern.sub(expand_match, text)
        expanded_text = re.sub("'", "", expanded_text)
        return expanded_text





    int_features = " ".join([str(word) for word in request.form.values()])
    final_features = normalize_corpus(int_features)
    vectorizer_input = vectorizer.transform([final_features])
    prediction = model.predict(vectorizer_input)

    return render_template('index.html',
                           prediction_text=("input was", int_features if prediction == 1 else 'negative')
                           )


if __name__ == "__main__":
    app.run(debug=True)
