# This script loads in the data and prints out some important statistics

# Imports
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("/Users/jswithers/personal-projects/nlp-practice/data/Reviews.csv")

corpus = df['Text'].copy()
target = df['Score']

def splitter(line):
    return line.split()

list_corpus = corpus.map(splitter)

cvec = CountVectorizer()

cvec.fit(corpus)
transformed = cvec.transform(corpus)

top_words = pd.DataFrame(transformed.toarray(), columns=cvec.get_feature_names())
