# -*- coding: utf-8 -*-
"""
Prepare data-set
Pre-processing (TF-IDF conversion, dimension reduction)
Train models with no hyper-parameter tuning
Train models with hyper-parameter tuning
"""
import os
import io
import pandas as pd
import preprocess_data
import models
from sklearn.model_selection import train_test_split


class ClassifyDoc(object):
    def __init__(self, data_path, model_path):
        self.data_path = data_path
        self.model_path = model_path

    def get_data(self):
        """
        Reads texts and creates data-frame
        """
        data_raw = []
        dirs = [label for label in os.listdir(self.data_path)
                if os.path.isdir(os.path.join(self.data_path, label))]
        for dir_n in dirs:
            dir_path = os.path.join(self.data_path, dir_n)
            text_paths = [os.path.join(dir_path, file_n) 
                          for file_n in os.listdir(dir_path)
                          if file_n.lower().endswith('txt')
                          and os.path.isfile(os.path.join(dir_path, file_n))]
            for text_path in text_paths:
                with io.open(text_path, "r") as f_in:
                    text = f_in.read()
                data_raw.append({'text': text, 'label': dir_n})
        data_frame = pd.DataFrame(data_raw)
        data_frame.dropna(inplace=True)
        return data_frame
  
    def data_conversion(self, x_train, x_test):
        """
        TF-IDF conversion
        """
        return preprocess_data.tfidf_conversion(
                                    x_train, x_test, self.model_path)
    
    def train(self):
        """
        Basic statistics of data
        Split train/test
        TF-IDF conversion
        Dimension reduction
        Train models with no hyper-parameter tuning
        Train models with grid search cross validation and select the best model
        """
        data = self.get_data()
        print("Summary of labels")
        print(data['label'].describe())
        
        x_tr, x_te, y_tr, y_te = train_test_split(data['text'], 
                                  data['label'], test_size=0.1,
                                  random_state=111, stratify=data['label'])
        train_tfidf, test_tfidf = self.data_conversion(x_tr, x_te)
 
        train_pca, test_pca = preprocess_data.pca_dim(
                              train_tfidf, test_tfidf)
        models.not_tuned_models(train_pca, test_pca, y_tr, y_te)

        models.best_model(train_tfidf.toarray(), test_tfidf.toarray(), 
                          y_tr, y_te, self.model_path)
