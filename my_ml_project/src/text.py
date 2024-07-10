import os
import sys
import itertools
from collections import Counter
import numpy as np
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_path)

import my_library.load_input_data as input_loader
import my_library.load_dictionary1 as dictionary1_loader 
import my_library.load_dictionary2 as dictionary2_loader 

import my_library.Validator as Validator

Validator_SVC = Validator.Validator()

X, y = Validator_SVC.load_data("../data/validation_preprocessed.pkl")
Validator_SVC.load_model("../models/model_C_1000.pkl")
from sklearn.metrics import confusion_matrix
print("SVC, C=1000:{}")
print(confusion_matrix(y, Validator_SVC.model.predict(X)))

Validator_MLP = Validator.Validator()
Validator_MLP.load_model("../models/model_MLP_neuron_1000_layer_100.pkl")
print("MLP, neuron=1000, layer=100")
print(confusion_matrix(y, Validator_MLP.model.predict(X)))