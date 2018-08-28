"""
Feature extraction for usernames. Designed to be used in conjunction with
other scikit-learn transformers.

The base class, FeatureExtractor, currently is set up to take arbitrary transformations
and return a matrix from a username feature.

The only child class currently available uses functions designed for Amazon food reviews.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import re

"""
Functions. These are used to map a username to a new numeric feature.
"""

def full_name_check(text):
    """
    Checks to see if a text matches the pattern First M. Last
    """
    pattern = r"\b[A-Z][a-z]+ [A-Z]. [A-Z][a-z]+"
    if re.match(pattern=pattern, string=str(text)) != None:
        return 1
    else:
        return 0

def vowel_counter(text):
    """
    Counts vowels in a text.
    """
    vowels = {'a', 'e', 'i', 'o', 'u'}
    count = 0
    for letter in str(text):
        if letter.lower() in vowels:
            count += 1
    return count

def odd_char_counter(text, check):
    """
    Counts artibrary chars in a text.
    """
    return sum(1 if char==check else 0 for char in text.lower() )

def x_counter(text):
    return odd_char_counter(text, 'x')
def y_counter(text):
    return odd_char_counter(text, 'y')
def z_counter(text):
    return odd_char_counter(text, 'z')
def q_counter(text):
    return odd_char_counter(text, 'q')
def v_counter(text):
    return odd_char_counter(text, 'v')

def upper_counter(s):
    """
    Counts the uppercase characters in a text.
    """
    return sum([1 if char.isupper() else 0 for char in str(s)])

common_male_names = set(['michael', 'christopher', 'joshua', 'matthew', 'david', 'daniel', 'andrew', 'joseph', 'justin', 'james', 'mike', 'chris','josh','matt','dave','dan','joe','john'])

"""
Classes. FeatureExtractor is the base class and AmazonFeatureExtractor is the only current child.
"""

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts several potentially important features from an Amazon username:
        - is_fullname
        - vowels_in_profilename
        - profile_length
        - profile_cap (is the profile capitalized?)
        - num_caps

    This implementation uses the sklearn BaseEstimator and TransformerMixin classes.
        """

    def fit(self):
        return None

    def transform(self, X):
        """
        transforms the text according to set steps.
        """
        X_copy = X.copy()
        X_copy = X_copy.values
        return X_copy

    def fit_transform(self, X, y=None):
        self.fit()
        return self.transform(X)

class AmazonFeatureExtractor(FeatureExtractor):

    def transform(self, X):
        X_copy = X.copy()

        X_copy['has_quotes'] = X_copy['ProfileName'].str.contains('"').map(lambda x: 1 if x else 0)
        X_copy['is_fullname'] = X_copy['ProfileName'].map(full_name_check)
        X_copy['vowels_in_profilename'] = X_copy['ProfileName'].map(vowel_counter)
        X_copy['profile_length'] = X_copy['ProfileName'].map(lambda x: len(str(x)))
        X_copy['profile_cap'] = X_copy['ProfileName'].map(lambda x: 1 if str(x)[0].isupper() else 0)
        X_copy['num_caps'] = X_copy['ProfileName'].map(upper_counter)
        X_copy['x_chars'] = X_copy['ProfileName'].map(x_counter)
        X_copy['y_chars'] = X_copy['ProfileName'].map(y_counter)
        X_copy['z_chars'] = X_copy['ProfileName'].map(z_counter)
        X_copy['q_chars'] = X_copy['ProfileName'].map(q_counter)
        X_copy['v_chars'] = X_copy['ProfileName'].map(v_counter)
        X_copy['n_words'] = X_copy['ProfileName'].map(lambda x: len(x.split()))
        X_copy['longest_word'] = X_copy['ProfileName'].map(lambda x: max([len(i) for i in x.split()]))
        X_copy['shortest_word'] = X_copy['ProfileName'].map(lambda x: min([len(i) for i in x.split()]))
        X_copy['contains_common_male_names'] = X_copy['ProfileName'].map(lambda x: sum([1 if i.lower() in common_male_names else 0 for i in x.split()]))

        self.feature_names_ = X_copy.columns
        X_copy = X_copy.drop(['ProfileName'], axis = 1).values
        return X_copy
