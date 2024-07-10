import os
import sys
import traceback  # Import traceback for detailed error reporting

print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_path)

import my_library.Validator as Validator

validator = Validator.Validator()

X, y = validator.load_data("../data/validation_preprocessed.pkl")

hyperparameters = [10,100,1000]

for C in hyperparameters:
    validator.load_model("../models/model_C_" + str(C) + ".pkl")
    accuracy, report, matrix = validator.evaluate_mode(X, y)
    print("C={}".format(C), accuracy)
    print(report)
    print(matrix)
    
    