A) SOURCES

The paper for this project as well as the table showing the features and models we have to do can be found in the following link
https://link.springer.com/article/10.1186/s13673-019-0205-6/tables/9

Codebase for this paper can also be found in the github link provided below
https://github.com/sumba101/HateSpeech

B) DATASETS

fulldataset.csv is the complete unaltered dataset given by the paper

cleanDataset.csv is the dataset produced after doing the preprocess on the text data, the code that was used for this preprocessing can be found as in a function in main.py

SimpleFeatures.csv is a csv of features made as per what the paper said. The code for making it can be found in simpleFeatures.ipynb

testingSet and trainingSet csvs are just the cleaned dataset split 

C) CODE

The code for training and saving the models on the Simple Features dataset can be found in simpleFeaturesModels.py

BOW and TFIDF and other pickles are pickled and saved vectorizer class models of the training set. I made it to use it in the pipeline format found in pipeline.py

Code for making this BOW and TFIDF models can be found in the testing.ipynb file

Code for the generation of bert,w2v and w2vtfidf can be found in respective python files named as such

The folders with model names have inside then text files that show the report card for the model using the features of said text file name
