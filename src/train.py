import os
import sys
import traceback  # Import traceback for detailed error reporting

print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_path)

import my_library.Trainer as Trainer

trainer = Trainer.Trainer()

try:
    X, y = trainer.load_data("../data/train_preprocessed.pkl")
    print("Data loaded successfully")

    hyperparameters = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    for h in hyperparameters:
        model_dump_path = "../models/model_C_" + str(h) + ".pkl"  # Corrected path
        print(f"Training with hyperparameter C={h}")
        
        trainer.train_SVM(h, X, y)
        print(f"Training completed for hyperparameter C={h}")
        
        trainer.dump_model(model_dump_path)
        print(f"Model saved to {model_dump_path}")
    
    print("Done training")
except Exception as e:
    print("Training Error")
    traceback.print_exc()  # Print the stack trace for the exception
