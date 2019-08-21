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
