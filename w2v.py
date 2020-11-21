import pandas as pd
import gensim
import pickle
import numpy as np

def save(name,obj):
    with open(name,'wb') as f:
        pickle.dump(obj,f)

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])

if __name__=="__main__":
    df = pd.read_csv("Datasets/trainingSet.csv")
    x = list(df["clean_text"])
    lines = [k.split(" ") for k in x]

    model = gensim.models.Word2Vec(lines, size=100) 
    w2v = dict(zip(model.wv.index2word, model.wv.vectors))

    vectorizer = MeanEmbeddingVectorizer(w2v)
    save("W2V",vectorizer)


