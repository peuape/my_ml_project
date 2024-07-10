import os
import sys
import traceback  # Import traceback for detailed error reporting

print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_path)

import my_library.Trainer_MLP as Trainer

trainer = Trainer.Trainer()

try:
    X, y = trainer.load_data("../data/train_preprocessed.pkl")
    X_scaled = trainer.Standard_Scaler(X)
    print("Data loaded and scaled successfully")
    
    trainer.dump_scaler("../models/standardscaler")
    print("Scaler saved to ../models/standardscaler")

    neurons, layers = [10, 100, 1000], [1, 10, 100]
    for neuron in neurons:
        for layer in layers:
            model_dump_path = "../models/model_MLP_neuron_" + str(neuron) +"_layer_"+str(layer)+ ".pkl"  # Corrected path
            print(f"Training with hyperparameter neuron={neuron}, layer={layer}")
        
            trainer.train_MLP(neuron, layer, X, y)
            print(f"Training completed for hyperparameter neuron={neuron}, layer={layer}")
        
            trainer.dump_model(model_dump_path)
            print(f"Model saved to {model_dump_path}")
    
    print("Done training")
except Exception as e:
    print("Training Error")
    traceback.print_exc()  # Print the stack trace for the exception
