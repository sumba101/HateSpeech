# reference : https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794
import pickle

import pandas as pd
import numpy as np


## for processing
import re
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

## for bag-of-words TFID and naive bayes
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing, metrics


## for word embedding
# import gensim
# import gensim.downloader as gensim_api
from sklearn.svm import LinearSVC


def utils_preprocess_text(text, flg_stemm=False, flg_lemm=True, lst_stopwords=None):
    '''
    Preprocess a string.
    :parameter
        :param text: string - name of column containing text
        :param lst_stopwords: list - list of stopwords to remove
        :param flg_stemm: bool - whether stemming is to be applied
        :param flg_lemm: bool - whether lemmitisation is to be applied
    :return
        cleaned text
    '''
    ## clean (convert to lowercase and remove punctuations and characters and then strip)
    text = re.sub( r'[^\w\s]', '', str( text ).lower().strip() )

    ## Tokenize (convert from string to list)
    lst_text = text.split()  ## remove Stopwords
    if lst_stopwords is not None:
        lst_text = [word for word in lst_text if word not in lst_stopwords]

    # Add section here to remove https links from the datapoints

    ## Stemming (remove -ing, -ly, ...)
    if flg_stemm == True:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem( word ) for word in lst_text]

    ## Lemmatisation (convert the word into root word)
    if flg_lemm == True:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize( word ) for word in lst_text]

    ## back to string from list
    text = " ".join( lst_text )
    return text


def preprocess(df):
    stopwords_list = nltk.corpus.stopwords.words( "english" )
    df["clean_text"] = df["Comment_text"].apply(
        lambda x: utils_preprocess_text( x, flg_stemm=False, flg_lemm=True, lst_stopwords=stopwords_list ) )
    df.to_csv( "cleanDataset.csv" )
    return df


def save(name,obj):
    with open(name,'wb') as f:
        pickle.dump(obj,f)
def load(name):
    with open(name,'rb') as f:
        return pickle.load(f)


# loads the dataset
df = pd.read_csv( "trainingSet.csv" )
x_train=df['clean_text']
y_train = df["Hateful_or_not"]
df2= pd.read_csv("testingSet.csv")
X_test=df2['clean_text']
y_test=df2['Hateful_or_not']

vectorizer = load("TFIDF")
print(type(vectorizer))

classifier = LinearSVC()
## pipeline
model1 = pipeline.Pipeline( [("vectorizer", vectorizer),
                                ("classifier", classifier)] )

model1["classifier"].fit( model1["vectorizer"].transform(x_train), y_train )

predicted = model1.predict( X_test )

print( metrics.classification_report( y_test, predicted ), file=open( "SVM/Tfidf.txt", 'w' ) )
