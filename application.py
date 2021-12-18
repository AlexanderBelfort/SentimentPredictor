import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from collections import Counter
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from contractions import contractions_dict
import re
import unicodedata
from bs4 import BeautifulSoup
import spacy
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
model = pickle.load(open("lr_model_final_gs.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

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

    def remove_stopwords(text, is_lower_case=False):
        tokenizer = ToktokTokenizer()
        tokens = tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]

        if is_lower_case:

            filtered_tokens = [token for token in tokens if token not in stopword_list]
        else:

            filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
        filtered_text = ' '.join(filtered_tokens)
        
        
        return filtered_text


    def lemmatize_text(text):
        text = nlp(text)
        return ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])

    def expand_contractions(text, contraction_mapping=contractions_dict):
        contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                          flags=re.IGNORECASE | re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]

            expanded_contraction = contraction_mapping.get(match) if contraction_mapping.get(
                match) else contraction_mapping.get(match.lower())
            return first_char + expanded_contraction[1:] if expanded_contraction != None else match

        expanded_text = contractions_pattern.sub(expand_match, text)

        expanded_text = re.sub("'", "", expanded_text)


        return expanded_text




    nltk.download('stopwords')
    stopword_list = nltk.corpus.stopwords.words('english')
    stopword_list.remove('no')
    stopword_list.remove('not')





    def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                         accented_char_removal=True, text_lower_case=True,
                         text_lemmatization=True, special_char_removal=True,
                         stopword_removal=True):

        # strip HTML
        if html_stripping:
            corpus = strip_html_tags(corpus)

        # remove accented characters
        if accented_char_removal:
            corpus = remove_accented_chars(corpus)

        # expand contractions
        if contraction_expansion:
            corpus = expand_contractions(corpus)

        # Lowercase the text
        if text_lower_case:
            corpus = corpus.lower()

        # remove extra newlines
        corpus = re.sub(r'[\r|\n|\r\n]+', ' ', corpus)

        # insert spaces between special characters to isolate them
        special_char_pattern = re.compile(r'([{.(-)!}])')
        corpus = special_char_pattern.sub(" \\1 ", corpus)

        # lemmatize text
        if text_lemmatization:
            corpus = lemmatize_text(corpus)

        # remove special characters
        if special_char_removal:
            corpus = remove_special_characters(corpus)

        # remove extra whitespace
        corpus = re.sub(' +', ' ', corpus)

        # remove stopwords
        if stopword_removal:
            corpus = remove_stopwords(corpus, is_lower_case=text_lower_case)


        return corpus


    features_toString = " ".join([str(word) for word in request.form.values()])
    str_input = normalize_corpus(features_toString)


    transformed_input_vectorizer = vectorizer.transform([str_input])
    pred = model.predict(transformed_input_vectorizer)

    return render_template('index.html', prediction_text=("SENTIMENT: positive." if (pred == 1) else "SENTIMENT: negative."))

if __name__ == "__main__":
    app.run(debug=True)
