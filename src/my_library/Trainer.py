from sklearn.svm import SVC
import pickle
import numpy as np

class Trainer:
    def __init__(self):
        self.model = None
        
    def load_data(self, file_path):
        with open(file_path, "rb") as f:
            X, y = pickle.load(f)
        return X, y
    
    def dump_model(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)
            
    def train_SVM(self, hyperparameter, X, y):
        self.model = SVC(C=hyperparameter)
        self.model.fit(X, y)