# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:12:20 2019

@author: sagar
"""
#importing necessary libraries
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import rcParams
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import explained_variance_score,median_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from flask import Flask, jsonify
from sklearn.externals import joblib
import pickle as pickle
from sklearn.base import BaseEstimator, TransformerMixin

class preprocessing(BaseEstimator, TransformerMixin):
    
        def __init__(self):
            pass
        
        #def add_interactions(self,df):
         #       combos = list(combinations(list(df.columns),2))
          #      colnames = list(df.columns) + ['_'.join(x) for x in combos]
           #     poly = PolynomialFeatures(interaction_only=True, include_bias= False)
         #       df = poly.fit_transform(df)
         #       df = pd.DataFrame(df)
         #       df.columns = colnames
         #       return df
        def transform(self, df):
                pred_var = ['num_of_cylinders', 'bore', 'fuel_system_2bbl',
       'num_of_cylinders_fuel_system_2bbl', 'engine_type_ohcv',
       'aspiration_turbo', 'bore_make_audi', 'num_of_cylinders_bore',
       'num_of_cylinders_symboling_1', 'bore_engine_type_ohcv', 'stroke',
       'make_volkswagen', 'symboling_2_make_volkswagen',
       'stroke_make_volkswagen',
       'make_mercedes_benz', 'make_saab', 'engine_type_ohcf', 'make_audi',
       'engine_location_rear', 'make_jaguar', 'symboling_1', 'make_bmw',
       'symboling_2']
                df.replace(to_replace='?',value=np.nan,inplace=True)
                df.dropna(subset=['num_of_doors','bore', 'stroke','horsepower', 'peak_rpm','price'],inplace=True)
                df['normalized_losses'].replace(to_replace=np.nan,value=0,inplace=True)
                df['normalized_losses'] = df['normalized_losses'].astype(int)
                df['normalized_losses'].replace(to_replace=0,value=df['normalized_losses'].mean(),inplace=True)
                df['num_of_doors'] = df['num_of_doors'].map({'four':4,'two':2})
                df['num_of_cylinders'] = df['num_of_cylinders'].map({'four':4,'six':6,'five':5,'eight':8,'three':3,'twelve':12})
                df['bore'] = df['bore'].astype(float)
                df['stroke'] = df['stroke'].astype(float)
                df['horsepower'] = df['horsepower'].astype(float)
                df['peak_rpm'] = df['peak_rpm'].astype(float)
                df['price'] = df['price'].astype(float)
                df['area'] = df['length'] * df['width'] * df['height']
                df['symboling'] = df['symboling'].astype(str)
                df.replace(regex=r'-',value='_',inplace=True)
                df.drop(['length','width','height'],axis=1,inplace=True)
                num_dtypes = df.select_dtypes(include=['int64','float64']).columns.values
                object_dtypes = df.select_dtypes(include=['object']).columns.values
                for i in object_dtypes :
                    df[i] = df[i].apply(lambda x : x.strip())
                for i in num_dtypes :
                    df[i] = np.log1p(df[i])
                scaler = StandardScaler()
                for j in num_dtypes :
                    df[j] = scaler.fit_transform(df[[j]])
                df_dummies = pd.get_dummies(df,columns=['symboling','make','fuel_type','aspiration','body_style','drive_wheels','engine_type','fuel_system','engine_location'],drop_first=True)
                df_dummies['Car_price'] = df_dummies['price']
                df_dummies.drop(['price'],axis=1,inplace=True)
                combos = list(combinations(list(df_dummies.columns),2))
                colnames = list(df_dummies.columns) + ['_'.join(x) for x in combos]
                poly = PolynomialFeatures(interaction_only=True, include_bias= False)
                df = poly.fit_transform(df_dummies)
                df = pd.DataFrame(df)
                df.columns = colnames
                #return df
                #df = add_interactions(df_dummies)
                #print(df)
                noint_indices = [i for i,x in enumerate(list((df==0).all())) if x]
                df = df.drop(df.columns[noint_indices],axis=1)
                df=df[pred_var]
                return df.as_matrix()
        def fit(self, df, y=None, **fit_params):
                """Fitting the Training dataset & calculating the required values from train
           e.g: We will need the mean of X_train['Loan_Amount_Term'] that will be used in
                transformation of X_test
                """  
                #self.term_mean_ = df['Loan_Amount_Term'].mean()
                #self.amt_mean_ = df['LoanAmount'].mean()
                return self
        
def build_and_train():
    df = pd.read_csv('C:\\Users\\sagar\\Documents\\automobile.csv',names=['symboling','normalized_losses','make','fuel_type','aspiration','num_of_doors','body_style','drive_wheels','engine_location','wheel_base','length','width','height','curb_weight','engine_type',
'num_of_cylinders','engine_size','fuel_system','bore','stroke','compression_ratio','horsepower','peak_rpm','city_mpg','highway_mpg','price'])
   
    pipe = make_pipeline(preprocessing(),SVR())
    print(pipe)
    param_grd = {'C':np.arange(0.01,10,0.05)}

    grid = GridSearchCV(pipe, param_grid=param_grd, cv=10)
    print('grid is ',grid)
    
    X_train, X_test, y_train, y_test = train_test_split(df[df.columns.difference(['price'])],df['price'], test_size=0.25, random_state=42)
    print('xtrain columns',X_train.columns)
    grid.fit(X_train, y_train)
    
    return(grid)
    

if __name__ == '__main__':
    model = build_and_train()
    model_columns = list(X_train1.columns)
    with open('C:/Users/sagar/Documents/automobile/model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print('dumped model!')
    with open('C:/Users/sagar/Documents/automobile/automobile_columns.pkl', 'wb') as file1:
        pickle.dump(model_columns, file1)
    print('dumped columns!!')

# Z = pd.read_csv('C:\\Users\\sagar\\Documents\\automobile.csv',names=['symboling','normalized_losses','make','fuel_type','aspiration','num_of_doors','body_style','drive_wheels','engine_location','wheel_base','length','width','height','curb_weight','engine_type',
#'num_of_cylinders','engine_size','fuel_system','bore','stroke','compression_ratio','horsepower','peak_rpm','city_mpg','highway_mpg','price'])
   
df = pd.read_csv('C:\\Users\\sagar\\Documents\\automobile.csv',names=['symboling','normalized_losses','make','fuel_type','aspiration','num_of_doors','body_style','drive_wheels','engine_location','wheel_base','length','width','height','curb_weight','engine_type',
'num_of_cylinders','engine_size','fuel_system','bore','stroke','compression_ratio','horsepower','peak_rpm','city_mpg','highway_mpg','price'])
   
pipe = make_pipeline(preprocessing(),SVR())
print(pipe)
param_grd = {'C':np.arange(0.01,10,0.05)}

grid = GridSearchCV(pipe, param_grid=param_grd, cv=10)
print('grid is ',grid)
    
X_train, X_test, y_train, y_test = train_test_split(df,df['price'], test_size=0.25, random_state=42)
print('xtrain columns',X_train.columns)
grid.fit(X_train, y_train)

k=preprocessing()
k.transform(X_train)
