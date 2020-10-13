Here are the explanations for the madness
A) SOURCES

The paper for this project as well as the table showing the features and models we have to do can be found in the following link
https://link.springer.com/article/10.1186/s13673-019-0205-6/tables/9

Codebase for this paper used by the author of the paper can be found below
https://github.com/joolsa/Binary-Classifier-for-Online-Hate-Detection-in-Multiple-Social-Media-Platforms
(it mainly contains shit about bert and xgboost tho)

https://github.com/joolsa/Binary-Classifier-for-Online-Hate-Detection-in-Multiple-Social-Media-Platforms/blob/master/Documentation%20Milestone%202.pdf

Small guide of sorts i used for doing the cleaning of the data, and structing of the TFIDF and BoW can be found here
https://towardsdatascience.com/text-classification-with-nlp-tf-idf-vs-word2vec-vs-bert-41ff868d1794

B) DATASETS

fulldataset.csv is the complete unaltered dataset given by the paper

cleanDataset.csv is the dataset produced after doing the preprocess on the text data, the code that was used for this preprocessing can be found as in a function in main.py

SimpleFeatures.csv is a csv of features made as per what the paper said. The code for making it can be found in simpleFeatures.ipynb

testingSet and trainingSet csvs are just the cleaned dataset split like this cos idk it made it easier and felt why not

C) CODE
The code for training and saving the SVM and Naive bayes models on the Simple Features dataset can be found in simpleFeaturesModels.py

BOW and TFIDF are pickled and saved vectorizer class models of the training set. I made it to use it in the pipeline format to make shit easier and quicker
	Code for making this BOW and TFIDF models can be found in the testing.ipynb file

main.py uses the TFIDF and BOW models after unpickling in a pipeline
	As of now i have made use of SVM and NaiveBayes in the pipeline and placed the output text files in the respective folders



The folders with model names have inside then text files that show the report card for the model using the features of said text file name

Do note that the results we are getting seem better than what the paper got as can be seen in the link below
https://link.springer.com/article/10.1186/s13673-019-0205-6/tables/9

I guess its cos we are preprocessing the text before training?? dont know. Do you think its a concern?

Work ahead:

Logistic regression and XGBoost are cute stuff only so if you want change the exist code up a little bit to run them and put off on ada. We could then add them to the interim report

Other than that of course we should also do the one you said you used in IRE assignment

