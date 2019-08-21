"""
Basic document classification
1- Collect texts from data folder
2- Pre-processing data
3- Model training
4- Result analysis
"""
import os
from classify import ClassifyDoc


if __name__ == "__main__":
    """
    Data path 
    Classification model path 
    Train classification model
    """
    code_path = os.path.dirname(os.path.realpath(__file__))
    base_path = os.path.dirname(code_path)
    data_path = os.path.join(base_path, 'data')
    classification_model_path = os.path.join(base_path, 'model')
    try:
        doc_class_obj = ClassifyDoc(data_path, classification_model_path)
        doc_class_obj.train()
    finally:
        del doc_class_obj
