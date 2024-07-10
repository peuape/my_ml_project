import os
import sys
import traceback  # Import traceback for detailed error reporting

print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_path)

import my_library.Validator as Validator

validator = Validator.Validator()

X, y = validator.load_data("../data/validation_preprocessed.pkl")

neurons, layers = [10, 100, 1000], [1, 10, 100]

for neuron in neurons:
    for layer in layers:
        validator.load_model("../models/model_MLP_neuron_" + str(neuron) +"_layer_"+str(layer)+ ".pkl")
        accuracy, report, matrix = validator.evaluate_mode(X, y)
        print("neuron={}".format(neuron), "layer={}".format(layer), accuracy)
        print(report)
        print(matrix)