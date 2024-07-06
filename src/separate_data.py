import os
import sys

print(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 

project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 

sys.path.append(project_root_path)

import my_library.load_input_data as input_loader




try:
    with open("../data/data.txt", 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            if "train" in line:
                with open("../data/train.txt", 'a', encoding='utf-8') as f:
                    f.write(line)
            elif "dev" in line:
                with open("../data/validation.txt", "a", encoding="utf-8") as f:
                    f.write(line)
            elif "test" in line:
                with open("../data/test.txt", "a", encoding="utf-8") as f:
                    f.write(line)
        print("Done separating")
except:
    print("Error")