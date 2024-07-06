import os
import sys
from collections import Counter
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root_path)

import my_library.load_input_data as input_loader



sentence_arrays = input_loader.load_text_from_file("../data/data.txt")
category_list = []
outlier_list = []
lengths = []
for i in range(len(sentence_arrays)):
    if len(sentence_arrays[i])<4:
        outlier_list.append(sentence_arrays[i])
        continue
    category_list.append(sentence_arrays[i][3])
    
    
for i in range(len(sentence_arrays)):
    lengths.append(len(sentence_arrays[i]))

print(Counter(category_list))
print(len(sentence_arrays))
print(outlier_list[:5])
print(Counter(lengths))