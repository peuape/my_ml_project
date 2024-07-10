import os
import sys

print(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 


project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 


sys.path.append(project_root_path)

import my_library.load_input_data as input_loader
import my_library.load_dictionary1 as dictionary1_loader 
import my_library.load_dictionary2 as dictionary2_loader 

from my_library.Data_preprocessor import DataPreprocessor


data_path = input("Enter the path for the data file: ")
dictionary1_path = input("Enter the path for the dictionary1 file: ") 
dictionary2_path = input("Enter the path for the dictionary2 file: ") 
preprocessed_data_path = input("Enter the path to save the preprocessed data: ")

sentence_arrays = input_loader.load(data_path) 
dictionary1 = dictionary1_loader.load(dictionary1_path)
dictionary2 = dictionary2_loader.load(dictionary2_path)
data_preprocessor = DataPreprocessor(sentence_arrays, dictionary1, dictionary2)
data_preprocessor.preprocess_and_dump(preprocessed_data_path)

print("Done preprocessing")