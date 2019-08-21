# -*- coding: utf-8 -*-
"""
Dimension reduction (Principal Component Analysis,
Linear Discriminant Analysis, Non-negative Matrix Factorization)
Convert text to array (TF-IDF)
"""
import os
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pickle as pk


def pca_dim(x_train, x_test, n_features=1000):
    pca = PCA(n_features)
    x_train = pca.fit_transform(x_train.toarray())
    x_test = pca.transform(x_test.toarray())
    return x_train, x_test


def lda_dim(x_train, x_test, y_train, n_features=1000):
    lda = LDA(n_components=n_features)
    x_train = lda.fit_transform(x_train.toarray(), y_train)
    x_test = lda.transform(x_test.toarray())
    return x_train, x_test


def nmf_dim(x_train, x_test, n_features=10):
    nmf = NMF(n_features)
    x_train = nmf.fit_transform(x_train.toarray())
    x_test = nmf.transform(x_test.toarray())
    return x_train, x_test


def tfidf_conversion(x_train, x_test, model_path):
    stop_words = set(stopwords.words("english"))
    tfidf = TfidfVectorizer(lowercase=True, analyzer='word', min_df=0.001,
                            max_df=0.3, ngram_range=(1, 3), norm='l2',
                            use_idf=True, stop_words=stop_words)
    tfidf_tr = tfidf.fit_transform(x_train)
    tfidf_te = tfidf.transform(x_test)
    pk.dump(tfidf, open(os.path.join(model_path, 'vectorized_model'), 'wb'))
    return tfidf_tr, tfidf_te
