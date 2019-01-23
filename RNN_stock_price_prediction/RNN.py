# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 19:40:28 2019

@author: Pathanjali
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
from sklearn.preprocessing import MinMaxScaler

#reading the dataset
train=pd.read_csv('C:\\Users\\sagar\\Downloads\\Recurrent_Neural_Networks-1\\Google_Stock_Price_Train.csv')
#see the plot of the open price
train['Open'].plot()
#create a numpy array for input to RNN
train_set=train.iloc[:,1:2].values
#scaling the data as we will be using sigmoid/relu functions
sc=MinMaxScaler(feature_range=(0,1))
#scaled the input data to range 0,1
train_set_scale=sc.fit_transform(train_set)

#initializing the lists for 60 previous stock prices and 1 output
X_train=[]

y_train=[]
for i in range(60,len(train_set_scale)):
    X_train.append(train_set_scale[i-60:i,0])
    y_train.append(train_set_scale[i,0])
X_train,y_train = np.array(X_train),np.array(y_train)

#adding more dimensions for predicting along with open price
#this creates a 3D numpy array
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))


#importing keras libraries and functions
from keras.models import Sequential
from keras.layers import Dense,LSTM,Dropout

#Initializing RNN
regressor= Sequential()

#adding LSTM and drop out regularization to avoid overfitting
#first iteration of layers
regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))
#second iteration of layers
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
#third iteration of layers
regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))
#fourth iteration of layers,setting it to false because there is no propagation
regressor.add(LSTM(units=50,return_sequences=False))
regressor.add(Dropout(0.2))
#adding the output layer
regressor.add(Dense(units=1))
#compiling RNN
regressor.compile(optimizer='adam',loss='mean_squared_error')
#fitting the model
regressor.fit(x=X_train,y=y_train,batch_size=32,epochs=100)

#importing the test dataset and open amount
test=pd.read_csv('C:\\Users\\sagar\\Downloads\\Recurrent_Neural_Networks-1\\Google_Stock_Price_Test.csv')
real_stock_price=test.iloc[:,1:2].values

#prediction requires past 60days of data. so we need to concatenate 
#train and test sets 
data_total=pd.concat((train['Open'],test['Open']),axis=0)
#to find the last 60 days of first financial day of jan2017
inputs=data_total[len(data_total)-len(test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)

#initializing the lists for 60 previous stock prices and 1 output
X_test=[]
for i in range(60,len(inputs)):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

#adding more dimensions for predicting along with open price
#this creates a 3D numpy array
X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock=regressor.predict(X_test)
predicted_stock=sc.inverse_transform(predicted_stock)

#visualizing the results
plt.plot(real_stock_price,color='red',label='real price')
plt.plot(predicted_stock,color='green',label='predicted price')
plt.title('google stock prediction')
plt.xlabel('Time')
plt.ylabel('stock price')
plt.legend()
plt.show()