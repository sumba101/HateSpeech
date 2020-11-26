import pandas as pd
import gensim
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def save(name,obj):
    with open(name,'wb') as f:
        pickle.dump(obj,f)

class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])

if __name__=="__main__":
    df = pd.read_csv("Datasets/unclean_train.csv", sep=";")
    x = list(df["Comment_text"])
    lines = [k.split(" ") for k in x]

    model = gensim.models.Word2Vec(lines, size=100) 
    w2v = dict(zip(model.wv.index2word, model.wv.vectors))

    vectorizer = TfidfEmbeddingVectorizer(w2v)
    save("W2VTFIDF",vectorizer)


