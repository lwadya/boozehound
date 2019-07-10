import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.metrics import pairwise_distances

class cocktail_recommender:
    '''
    Builds an NMF cocktail recommendation engine from a DataFrame of drink
    information and two lists of safe words. The __init__ function sets up the
    engine and the recommend function runs it.
    '''

    # Default settings for pairwise distance metric and weight of search-matched
    # drink vector
    dist_metric = 'cosine'
    drink_weight = .4

    def __init__(self, df, full_stop_words, limited_stop_words):
        '''
        Saves DataFrame and stop word lists as class variables and uses them to
        create TFIDF Matrix and NMF vectors for both lists of stop words

        Args:
            df (DataFrame): cocktail recipe data
            full_stop_words (set): stop words for fun model
            limited_stop_words (set): stop words for safe model

        Returns:
            None
        '''
        self.df = df

        # Creates safe model word vectors
        self.safe_tfidf = TfidfVectorizer(stop_words=full_stop_words)
        self.safe_mtx = self.safe_tfidf.fit_transform(df['description'].values)
        self.safe_nmf = NMF(n_components = 30)
        self.safe_drink_vec = self.safe_nmf.fit_transform(self.safe_mtx)
        self.safe_word_vec = self.safe_nmf.components_.transpose()

        # Creates fun model word vectors
        self.fun_tfidf = TfidfVectorizer(stop_words=limited_stop_words)
        self.fun_mtx = self.fun_tfidf.fit_transform(df['description'].values)
        self.fun_nmf = NMF(n_components = 25)
        self.fun_drink_vec = self.fun_nmf.fit_transform(self.fun_mtx)
        self.fun_word_vec = self.fun_nmf.components_.transpose()

        # Creates a series of sets representing drink names
        self.name_sets = self.df['name_words'].apply(set)

    def recommend(self, input_string, weirdness=.5, num_recos=10,
                  exclude_inputs=False):
        '''
        Recommends a set number of recipes based on an input string

        Args:
            input_string (str): search string for recommendations
            weirdness (float): a blend between the safe and fun models, 0 is
                               entirely safe and 1 entirely fun
            num_recos (int): number of recommendations to return
            exclude_inputs (bool): if True removes drinks that match search
                                   string from the recommendations

        Returns:
            bool: True if model finds recommendations, False if not
            DataFrame: Recommended recipe data
        '''
        if not input_string:
            return False, None
        weirdness = max(min(weirdness, 1), 0)
        name_set = search_set = self.clean_input(input_string)
        drink_idx = self.name_matches(name_set)

        # Calculates safe search vector
        safe_search_vec = (self.safe_nmf.transform(
                           self.safe_tfidf.transform(search_set)))
        safe_search_vec = np.mean(safe_search_vec, axis=0, keepdims=True)

        # Calculates fun search vector
        fun_search_vec = (self.fun_nmf.transform(
                          self.fun_tfidf.transform(search_set)))
        fun_search_vec = np.mean(fun_search_vec, axis=0, keepdims=True)

        # Averages search vectors with matched drink vectors
        if drink_idx:
            safe_drink_vec = np.mean(self.safe_drink_vec[[drink_idx]],
                                     axis=0,
                                     keepdims=True)
            fun_drink_vec = np.mean(self.fun_drink_vec[[drink_idx]],
                                    axis=0,
                                    keepdims=True)
            safe_search_vec = ((1 - self.drink_weight) * safe_search_vec +
                               self.drink_weight * safe_drink_vec)
            fun_search_vec = ((1 - self.drink_weight) * fun_search_vec +
                               self.drink_weight * fun_drink_vec)

        # Calculates pairwise distances between search vectors and
        # recommendations
        if not safe_search_vec.sum():
            return False, None
        elif not fun_search_vec.sum():
            dist = pairwise_distances(X=self.safe_drink_vec,
                                      Y=safe_search_vec,
                                      metric=self.dist_metric)
        else:
            safe_dist = pairwise_distances(X=self.safe_drink_vec,
                                           Y=safe_search_vec,
                                           metric=self.dist_metric)
            fun_dist = pairwise_distances(X=self.fun_drink_vec,
                                          Y=fun_search_vec,
                                          metric=self.dist_metric)
            dist = (1 - weirdness) * safe_dist + weirdness * fun_dist

        # Calculates recommendations
        rank_idx = dist.transpose()[0].argsort().tolist()
        if exclude_inputs:
            rank_idx = list(filter(lambda x: x not in drink_idx, rank_idx))

        return True, self.df.loc[rank_idx].head(num_recos)

    def clean_input(self, input_string):
        '''
        Standardizes input text and converts it into a set of words

        Args:
            input_string (str): search string for recommendations

        Returns:
            set: cleaned up search words
        '''
        clean_str = re.sub('[^a-z0-9 \-]', '', input_string.lower().strip())
        return set(clean_str.split())

    def name_matches(self, input_set):
        '''
        Finds all drink names that match input words

        Args:
            input_set (set): cleaned up search words

        Returns:
            string list: matching drink names
        '''
        mask = (self.name_sets - input_set) == set()
        return self.name_sets[mask].index.tolist()
