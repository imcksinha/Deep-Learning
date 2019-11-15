# Artificial Nwural Networks

# Installing Theano

# Installing Tensorflow

# Installing Keras

# PART 1 - DATA PREPARATION
import os
os.getcwd()
os.chdir('C:/Chandan/Deep Learning/16_page_p0s1_file_1\Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning\Part 1 - Artificial Neural Networks (ANN)/Section 4 - Building an ANN')

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
dataset.shape # (10000, 14)
pd.set_option('display.max_columns', 14)
dataset.head()
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:] 

# Splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 0)

# Adding a new record to the test set for homework
X_test_new = np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])
X_test = np.append(X_test,X_test_new,axis=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)



# PART 2 - BUILDING ANN

# Import the required libraries
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initializing the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 ))
classifier.add(Dropout(p=0.1))  # To reduce overfitting
#classifier.add(Dense(activation="relu", input_dim=11, units=6, kernel_initializer="uniform"))

# Adding the second hidden layer
classifier.add(Dense(units=6, activation='relu',kernel_initializer='uniform'))
classifier.add(Dropout(p=0.1)) # To reduce overfitting

# Adding the output layer
classifier.add(Dense(units=1,activation='sigmoid', kernel_initializer='uniform'))

# Compiling the ANN
classifier.compile(optimizer='adam',loss = 'binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
classifier.fit(X_train,y_train,batch_size=10, epochs = 100)

# Predicting the test values
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

# Evaluating the ANN model through k fold cross validation
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11 ))
    classifier.add(Dense(units=6, activation='relu',kernel_initializer='uniform'))
    classifier.add(Dense(units=1,activation='sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer='adam',loss = 'binary_crossentropy', metrics=['accuracy'])
    return classifier
    
classifier = KerasClassifier(build_fn = build_classifier, batch_size=10, epochs = 100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
mean = accuracies.mean()#0.8413749952986838
variance = accuracies.std()#0.01685832430259701

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense

def build_classifier(optimizer,hidden_layer_neurons):
    classifier = Sequential()
    classifier.add(Dense(output_dim = hidden_layer_neurons, init = 'uniform', activation = 'relu', input_dim = 11 ))
    classifier.add(Dense(units=hidden_layer_neurons, activation='relu',kernel_initializer='uniform'))
    classifier.add(Dense(units=1,activation='sigmoid', kernel_initializer='uniform'))
    classifier.compile(optimizer = optimizer,loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
    
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[20,25],
              'hidden_layer_neurons':[5,6,8,10],
              #'epochs':[500, 800],
              'epochs':[200],
              'optimizer':['adam', 'rmsprop','sgd','adamax'],
              #'kernel_initializer':['uniform','lecun_uniform','normal','zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
              }
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train,y_train)
best_parameter = grid_search.best_params_
best_accuracy = grid_search.best_score_