# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 19:16:28 2020

@author: Nour

isnpried from https://github.com/dipanjanS/practical-machine-learning-with-python/blob/master/notebooks/Ch07_Analyzing_Movie_Reviews_Sentiment/text_normalizer.py
modifications:
"""

#
from bs4 import BeautifulSoup
import unicodedata
import re
from contractions import CONTRACTION_MAP
import spacy
import nltk
from nltk.tokenize.toktok import ToktokTokenizer

contractions_pattern = re.compile('({})'.format('|'.join(CONTRACTION_MAP.keys())),
                                      flags=re.IGNORECASE|re.DOTALL)

nlp = spacy.load("en_core_web_sm", disabled=["parser", "tagger","ner"])
tokenizer = ToktokTokenizer()
nltk.data.path.append('./nltk_data/')
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')


def strip_html(doc):
    soup = BeautifulSoup(doc, 'html.parser')
    return soup.get_text()

def remove_accented_chars(doc):
    return unicodedata.normalize('NFKD', doc).encode('ascii', 'ignore').decode('utf-8', 'ignore')

def expand_contractions(text):

    def expand_match(contraction):
        match = contraction.group(0)
        return CONTRACTION_MAP.get(match.lower())

    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text

def remove_special_characters(text):
    text = re.sub('[^a-zA-Z\s]+', '', text)
    return text

def lemmatize(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def tokinize(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    return tokens

def remove_stopwords(text, is_lower_case=False):
    tokens = tokinize(text)
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True,
                     text_lemmatization=True, special_char_removal=True,
                     stopword_removal=True):

    normalized_corpus = []
    n_docs = corpus.size
    i = 0
    for doc in corpus:
        i += 1
        percentage = (i/n_docs)*100
        print("\rnormalizing {:.2f}%".format(percentage), end = '')
        if html_stripping:
            doc = strip_html(doc)

        if accented_char_removal:
            doc = remove_accented_chars(doc)

        if contraction_expansion:
            doc = expand_contractions(doc)

        if text_lower_case:
            doc = doc.lower()

        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # insert spaces between special characters to isolate them
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)

        if text_lemmatization:
            doc = lemmatize(doc)

        if special_char_removal:
            doc = remove_special_characters(doc)

        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)

        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)

        normalized_corpus.append(doc)

    return normalized_corpus
