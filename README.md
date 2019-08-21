# Document-Classification
![](https://github.com/seanaba/Document-Classification/blob/master/doc/pic/pic1.jpg)
## **Table of Contents**
- [Introduction](#intro)
- [Classification Data](#cd)
- [Feature Extraction](#fe)
- [Dimensionality Reduction](#dr)
- [Document Classification Methods](#cm)
- [Evaluation Metrics](#em)
<a name="intro"></a>
## Introduction
Natural Language Processing (NLP) is growing drastically since millions of documents are generated every single day. Document classification is applying machine learning methods to classify documents into categories. It is supervised learning method whereas data tagging is necessary to train classification models. Document classification has wide variety of applications such as spam filtering, sentiment analysis and many other applications.  
Document classification can be simple or complicated depending on data, level of accuracy acceptance, language and many other parameters. Basic document classification method is presented in the following and it can be modified easily to cover more complicated cases. 
First, data used to evaluate models is addressed. Data pre-processing including feature extraction and dimensionality reduction is explained afterwards. Finally, model training and its evaluation are described in the last sections. 
<a name="cd"></a>
## Classification Data
As mentioned above, tagging data is necessary for document classification. Document tagging is time consuming process depending on size of documents required for model training as well as number of categories which documents are assigned to. There are variety of free resources available online. In this repository BBC News data is used to train and evaluate classification models. The dataset consists of 5 categories including business, entertainment, politics, sport, and technology. Total of 2225 documents from the BBC news website are provided and can be downloaded [here]( http://mlg.ucd.ie/datasets/bbc.html).
<a name="fe"></a>
## Feature Extraction
Feature extraction for document classification requires pre-processing. First of all, data is inspected briefly to check basic statistics and it can be expanded by using visualization tools as well as statistical tests. 
```python
print(data['label'].describe())
count      2225
unique        5
top       sport
freq        511
```
Since we are dealing with texts, text pre-processing such as removing stop words, spelling correction, stemming, and lemmatization can improve final results. For example, NLTK library provides many pre-processing tools. Other libraries such as Gensim and Spacy are useful for text pre-processing.
```python
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
```
The next step is to convert text/word to array of numbers to feed classification models. Depending on classification models to be used for training classification model as well as data characteristics the following conversion methods for text/word can be used in the pre-processing. Term Frequency – Inverse Document Frequency (TF-IDF) is used to convert text to array in this repository.
-	One-hot vector
-	Term Frequency
-	Term Frequency – Inverse Document Frequency 
-	Global Vectors for Word Representation
-	Word2vec
-	FastText
-	Contextualized Word Representations
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(lowercase=True, analyzer='word', min_df=0.001,
                            max_df=0.3, ngram_range=(1, 3), norm='l2',
                            use_idf=True, stop_words=stop_words)
tfidf_tr = tfidf.fit_transform(x_train)
tfidf_te = tfidf.transform(x_test)
```
<a name="dr"></a>
## Dimensionality Reduction
Dimensionality reduction methods are used to reduce number of features. As a result, it yields shorter processing time for model training and more importantly, better results of the trained models. Various methods are proposed to reduce dimension of features as a few of those methods are mentioned in the following. Principal Component Analysis (PCA) is used in this repository to reduce features’ dimension.
-	Principal Component Analysis 
-	Non-negative Matrix Factorization
-	Linear Discriminant Analysis
-	T-distributed Stochastic Neighbor Embedding
-	Auto-encoder
```python
from sklearn.decomposition import PCA
pca = PCA(n_features)
x_train = pca.fit_transform(x_train.toarray())
x_test = pca.transform(x_test.toarray())
```
The data pre-processing is not done here. Visualization of data and statistical test such chi-squared test to find the correlation/auto-correlation of data are other available tools to improve features and final results.
<a name="cm"></a>
## Document Classification Methods



