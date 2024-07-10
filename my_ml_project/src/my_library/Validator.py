import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
import joblib
import pickle
import os
import sys

print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_path)

import my_library.Trainer_SVM as Trainer

trainer = Trainer.Trainer()

class Validator:
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def load_data(self, file_path):
        with open(file_path, "rb") as f:
            X, y = pickle.load(f)
        return X, y
    
    def load_model(self, file_path):
        with open(file_path, "rb") as f:
            self.model = pickle.load(f)
            
    
            
    def evaluate_mode(self, X, y):
        with open("../models/standardscaler", "rb") as f:
            self.scaler = joblib.load(f)
        X_scaled = self.scaler.transform(X)    
        y_pred = self.model.predict(X_scaled)
        accuracy = metrics.accuracy_score(y, y_pred)
        report = metrics.classification_report(y, y_pred)
        matrix = confusion_matrix(y, y_pred)
        return accuracy, report, matrix
    
    