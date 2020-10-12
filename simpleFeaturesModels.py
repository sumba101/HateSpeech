import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

df = pd.read_csv( "SimpleFeatures.csv")
y = df.Hateful_or_not
copy=df.drop(["Hateful_or_not","Comment_text","Platform"],axis=1)

# using only the simple features


X_train, X_test, y_train, y_test = train_test_split( copy, y, test_size=0.30 )
svmClassifier = SVC(kernel='linear')
svmClassifier.fit(X_train,y_train)
y_pred=svmClassifier.predict(X_test)

with open( 'SVM/Simple.txt', 'a' ) as f:
    print( confusion_matrix( y_test, y_pred ), file=f )
    print( classification_report( y_test, y_pred ), file=f )

naiveBayesClassifier = MultinomialNB()
naiveBayesClassifier.fit(X_train,y_train)
y_pred2=naiveBayesClassifier.predict(X_test)

with open( 'NaiveBayes/Simple.txt', 'a' ) as f:
    print( confusion_matrix( y_test, y_pred ), file=f )
    print( classification_report( y_test, y_pred ), file=f )
