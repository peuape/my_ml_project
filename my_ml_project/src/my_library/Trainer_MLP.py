from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import numpy as np

class Trainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        
    def load_data(self, file_path):
        with open(file_path, "rb") as f:
            X, y = pickle.load(f)
        return X, y
    
    def Standard_Scaler(self, X):
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def dump_model(self, file_path):
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)
            
    def dump_scaler(self, file_path):
        
        with open(file_path, "wb") as f:
            joblib.dump(self.scaler, f)
            
    def train_MLP(self, neuron, layer, X, y):
        self.model = MLPClassifier(solver="lbfgs", random_state=0, hidden_layer_sizes=[neuron, layer])
        self.model.fit(X, y)