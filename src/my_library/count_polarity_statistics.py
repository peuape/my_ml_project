# d1, d2: type: dictionary
# e.g. d1: {"僕": 0.0, "くそ": -1.0, ...} 
#      d2: {"嬉しい": 1.0, "怒る": -1.0, ...}
# sentence: type: list 
# e.g. ["僕", "は", "嬉しい", "です"]
import MeCab
import numpy as np

def counter(d1, d2, sentence):
    # Since only dictionary 1 (d1) is used, the function should take two arguments.
    count_statistics = []
    for word in sentence:
        if word in d1:
            count_statistics.append(d1[word])
        elif word in d2:
            count_statistics.append(d2[word])
        else:
            count_statistics.append(0.0)  # If the word is not in either dictionary, append 0.
    return count_statistics  # e.g. [0.0, 0.0, 1.0, 0.0]


def count_and_vectorize(d1, d2, sentence):
    mecab = MeCab.Tagger("-Owakati")
    processed_sentence = [mecab.parse(sentence).strip().replace(' ', ',')]
    count_statistics = counter(d1, d2, processed_sentence)
    p = count_statistics.count(1.0)
    n = count_statistics.count(-1.0)
    e = count_statistics.count(0.0)
    vector = [p, n, e, p+n+e]
    return vector
    
    
    