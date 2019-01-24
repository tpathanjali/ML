# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 19:40:28 2019

@author: Pathanjali
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime as dt

#need to preprocess properly 

from sklearn.preprocessing import MinMaxScaler

#reading the dataset
train=pd.read_csv('C:\\Users\\sagar\\Downloads\\Recurrent_Neural_Networks-1\\Google_Stock_Price_Train.csv')
#see the plot of the open price
train['Open'].plot()
#close price has comma in the value so replaced them and converted them to float
train["Close"] = train["Close"].str.replace(",","").astype(float)
#convert the date to proper date format
train['Date'] = pd.to_datetime(train['Date'], format='%m/%d/%Y').dt.strftime('%m/%d/%Y')
#this function gives 1 as output for friday(weekend) else 0 for other days
def day_month(y):
    month, day, year = (int(x) for x in y.split('/'))    
    ans = dt.date(year, month, day)
    if ans.strftime("%a") == 'Fri':
        return 1
    else:
        return 0
def quarter(y):
    month, day, year = (int(x) for x in y.split('/'))    
    if month <=3:
        return 1
    elif month <=6:
        return 2
    elif month <=9:
        return 3
    else:
        return 4
    
#apply the friday indicator function to the dataset
train['friday_ind']=train['Date'].apply(day_month)
train['quarter']=train['Date'].apply(quarter)
train=pd.get_dummies(train,columns=['quarter'],drop_first=True)


#create a numpy array for input to RNN
train_set=train.iloc[:,[1,4,6,7,8,9]].values
#scaling the data as we will be using sigmoid/relu functions
sc_x=MinMaxScaler(feature_range=(0,1))
sc_y=MinMaxScaler(feature_range=(0,1))
#scaled the input data to range 0,1
#train_set_x=sc_x.fit_transform(train_set[0])

#initializing the lists for 60 previous stock prices and 1 output
X_train=[]
y_train=[]
for i in range(60,len(train_set)):
    X_train.append(train_set[i-60:i,0])
    y_train.append(train_set[i,1])
extra = train_set[60:, 1:].reshape(1198,5)
X_train,y_train = np.array(X_train),np.array(y_train)
X_train = np.hstack((X_train,extra))

train_set_x=sc_x.fit_transform(X_train)
train_set_y=sc_y.fit_transform(y_train.reshape(-1,1))
#adding more dimensions for predicting along with open price
#this creates a 3D numpy array
X_train=np.reshape(train_set_x,(train_set_x.shape[0],train_set_x.shape[1],1))
y_train=train_set_y

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
regressor.fit(x=X_train,y=y_train,batch_size=32,epochs=10)

#importing the test dataset and open amount
test=pd.read_csv('C:\\Users\\sagar\\Downloads\\Recurrent_Neural_Networks-1\\Google_Stock_Price_Test.csv')
test['Date'] = pd.to_datetime(test['Date'], format='%m/%d/%Y').dt.strftime('%m/%d/%Y')
test['friday_ind']=test['Date'].apply(day_month)
test['quarter']=test['Date'].apply(quarter)

test=pd.get_dummies(test,columns=['quarter'],drop_first=True)

real_stock_price=test.iloc[:,[4]].values

#prediction requires past 60days of data. so we need to concatenate 
#train and test sets 
data_total=pd.concat((train,test),axis=0)
#close price has comma in the value so replaced them and converted them to float
#data_total["Close"] = data_total["Close"].str.replace(",","").astype(float)
#to find the last 60 days of first financial day of jan2017
inputs=data_total[len(data_total)-len(test)-60:]
inputs=inputs.drop(['Date','Volume','High','Low'],axis=1)
inputs=inputs.fillna(0)
inputs=inputs.values

#inputs=sc_x.transform(inputs)

#initializing the lists for 60 previous stock prices and 1 output
X_test = []
for i in range(60, len(inputs)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
extra = inputs[60:, 1:]
X_test = np.hstack((X_test,extra))
X_test_scaled = sc_x.transform(X_test)
X_test_scaled = np.reshape(X_test_scaled,(X_test_scaled.shape[0],X_test_scaled.shape[1],1))

#adding more dimensions for predicting along with open price
#this creates a 3D numpy array
#X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
predicted_stock=regressor.predict(X_test_scaled)
#predicted_stock=np.append(X_test_scaled[:,0,0],predicted_stock)
#predicted_stock=predicted_stock.reshape(-2,2)
predicted_stock=sc_y.inverse_transform(predicted_stock)

#visualizing the results
plt.plot(real_stock_price,color='red',label='real price')
plt.plot(predicted_stock,color='green',label='predicted price')
plt.title('google stock prediction')
plt.xlabel('Time')
plt.ylabel('stock price')
plt.legend()
plt.show()