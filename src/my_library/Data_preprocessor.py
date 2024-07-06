import os
import sys
import pickle
import numpy as np
import my_library.count_polarity_statistics as counter

class DataPreprocessor:
    
    def __init__(self, sentence_arrays, dictionary1, dictionary2):
        self.sentence_arrays = sentence_arrays
        self.dictionary1 = dictionary1
        self.dictionary2 = dictionary2
        
    def preprocess_and_dump(self, file_path):

        X_list, y_list = [], []
        for i in range(len(self.sentence_arrays)):
            sentence = self.sentence_arrays[i][0] # Sentenceの 列 を 取 得
            count_vector = counter.count_and_vectorize(self.dictionary1, self.dictionary2, sentence) 
            X_list.append(count_vector)
            if self.sentence_arrays[i][4] == "0": # Writer_Joyの 列 が"0"な ら 負 例 と す る
                y_list.append(-1)
            else: # Writer_Joyの 列 が"0"で な い な ら 正 例 と す る
                y_list.append(1)
        X, y = np.array(X_list), np.array(y_list) # リ ス ト をnumpyの 多 次 元 配 列 に 変 換
        with open(file_path, "wb") as f:
            pickle.dump((X, y), f)
            