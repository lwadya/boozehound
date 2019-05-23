import re
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

class cocktail_recommender:
    dist_metric = 'cosine'
    drink_weight = .4

    def __init__(self):
        # Connects important data structures to class variables
        self.df = df
        #self.scy = scy
        self.safe_tfidf = safe_tfidf
        self.safe_nmf = safe_nmf
        self.safe_drink_vec = safe_drink_vec
        self.fun_tfidf = fun_tfidf
        self.fun_nmf = fun_nmf
        self.fun_drink_vec = fun_drink_vec

        # Creates a series of sets representing drink names
        self.name_sets = self.df['name_words'].apply(set)

    # Recommend function
    def recommend(self, input_string, weirdness=.5, num_recos=10, exclude_inputs=False):
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

        # Calculates pairwise distances between search vectors and recommendations
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

    # Cleans input string for title match and NMF vectorization
    def clean_input(self, input_string):
        clean_str = re.sub('[^a-z0-9 \-]', '', input_string.lower().strip())
        #doc = self.scy(clean_str)
        #words = [token.lemma_ for token in doc]
        return set(clean_str.split())#, set(words)

    # Find cocktail name matches in input set
    def name_matches(self, input_set):
        mask = (self.name_sets - input_set) == set()
        return self.name_sets[mask].index.tolist()
