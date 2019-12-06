import time
import numpy as np
from sklearn.externals import joblib
from time import perf_counter

############################ KNN ######################################
# Read in data
import pandas as pd
df = pd.read_csv('./data/data - Copy.csv')
# df = pd.read_csv('chroma_1.csv')
df = pd.DataFrame(df)
# print(df)
label = list(df['label'])
feature1 = list(df['C '])
feature2 = list(df['C#'])
feature3 = list(df['D '])
feature4 = list(df['D#'])
feature5 = list(df['E '])
feature6 = list(df['F '])
feature7 = list(df['F#'])
feature8 = list(df['G '])
feature9 = list(df['G#'])
feature10 = list(df['A '])
feature11 = list(df['A#'])
feature12 = list(df['B '])

# Import LabelEncoder
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
# Converting features into numbers
feature1_encoded = feature1
feature2_encoded = feature2
feature3_encoded = feature3
feature4_encoded = feature4
feature5_encoded = feature5
feature6_encoded = feature6
feature7_encoded = feature7
feature8_encoded = feature8
feature9_encoded = feature9
feature10_encoded = feature10
feature11_encoded = feature11
feature12_encoded = feature12
# Converting label into numbers
label_encoded = le.fit_transform(label)

#combinig features into single list of tuples
features=list(zip(feature1_encoded,feature2_encoded,feature3_encoded,feature4_encoded,feature5_encoded,feature6_encoded,feature7_encoded,feature8_encoded,feature9_encoded,feature10_encoded,feature11_encoded,feature12_encoded))
print(label_encoded.shape)

print('one sample: {}'.format(features[100]))
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(features, label_encoded, test_size=0.3) # 70% training and 30% test
print('one train sampe: {}'.format(X_train[100]))
start = perf_counter()
# Import knearest neighbors Classifier model
from sklearn.neighbors import KNeighborsClassifier
# Create KNN Classifier
knn = KNeighborsClassifier(n_neighbors=7)
# Train the model using the training sets
knn.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = knn.predict(X_test)

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("KNN Accuracy:",metrics.accuracy_score(y_test, y_pred))
elapsed = (perf_counter() - start)
print("KNN Time used:",elapsed)


############################ Sometimes Naive ######################################
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(features, label_encoded, test_size=0.3) # 70% training and 30% test

start = perf_counter()
# Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
# Create a Gaussian Classifier
gnb = GaussianNB()
# Train the model using the training sets
gnb.fit(X_train, y_train)
# Predict the response for test dataset
y_pred = gnb.predict(X_test)
# save model
joblib.dump(gnb, './model/gnb_mine.pkl')
print('model saved as '+'./model/gnb_mine.pkl')

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Gaussian NB Accuracy:",metrics.accuracy_score(y_test, y_pred))
elapsed = (perf_counter() - start)
print("NB Time used:",elapsed)

############################ SVM ######################################
# Import train_test_split function
from sklearn.model_selection import train_test_split
# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(features, label_encoded, test_size=0.3) # 70% training and 30% test

# Import svm model
from sklearn import svm
# Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel
# Train the model using the training sets
clf.fit(X_train, y_train)
start = perf_counter()
# Predict the response for test dataset
y_pred = clf.predict(X_test)
elapsed = (perf_counter() - start)
print("SVM Time used:",elapsed)

# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy: how often is the classifier correct?
print("SVM Accuracy:",metrics.accuracy_score(y_test, y_pred))
# joblib.dump(clf, "clf_train_model.m")
