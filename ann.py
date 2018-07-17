# Artificial Neural Network


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

# Encoding categorical data -> dummy variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#Convert categorical into numerical
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#Remove one dummy variable to avoid the dummy variable trap
X = X[:, 1:]

#Dividing the dataset into test and train
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0) 

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Import Keras Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense


classifier = Sequential()

#add input layer and hidden layer
classifier.add(Dense(units=6,activation="relu",input_dim=11,kernel_initializer="uniform")) #Selected 6 nodes in the hidden layer ; 
#Uniform distribution of weights ; selected activation rectification for hidden layers
#Input dim =11 independent variabels

#adding second hidden layer
classifier.add(Dense(units=6,activation="relu",kernel_initializer="uniform")) #Selected 6 nodes in the hidden layer ; 

#add output layer
classifier.add(Dense(units=1,activation="sigmoid",kernel_initializer="uniform"))

#COmpiling and adding S.G.D
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) #Using log loss function
#use categorical_crossentropy if more than 2 classifications 

#Fit into ANN
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)
#Batch size tells after how many inputs you need to update the weights
#Number of epochs = total iterations for a batch

#totak accuracy after convergence = 84%

#Prediction for test set
y_pred=classifier.predict(X_test) #in terms of probablities
#deciding a threshold for probs
y_pred=(y_pred>0.5) #returns true if >0.5

#confusion Matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#Accuracy:84%