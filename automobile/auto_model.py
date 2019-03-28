# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:12:20 2019
SVR was chosen after performing all types of regression 
on the data. SVR performed the best. 
A grid search was done and best parameters were taken out and hard coded here
@author: sagar
"""
#importing necessary libraries
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
import dill as pickle #dill has advantage to serialize custom transformers
from sklearn.base import BaseEstimator, TransformerMixin

class preprocessing(BaseEstimator, TransformerMixin):
        """BaseEstimator, TransformerMixin helps in inheriting fit method
        without actually doing anything
        """
        def __init__(self):
            pass
        
        def transform(self, df):
                pred_var = ['num_of_cylinders', 'bore', 'fuel_system_2bbl', 'engine_type_ohcv',
       'make_saab', 'engine_type_ohcf', 'aspiration_turbo', 'stroke',
       'make_audi', 'engine_location_rear', 'make_jaguar',
       'make_mercedes_benz', 'symboling_1', 'make_bmw', 'symboling_2',
       'make_volkswagen']
                df.replace(to_replace='?',value=np.nan,inplace=True)
                df.dropna(subset=['num_of_doors','bore', 'stroke','horsepower', 'peak_rpm'],inplace=True)
                df['normalized_losses'].replace(to_replace=np.nan,value=0,inplace=True)
                df['normalized_losses'] = df['normalized_losses'].astype(int)
                df['normalized_losses'].replace(to_replace=0,value=df['normalized_losses'].mean(),inplace=True)
                df['num_of_doors'] = df['num_of_doors'].map({'four':4,'two':2})
                df['num_of_cylinders'] = df['num_of_cylinders'].map({'four':4,'six':6,'five':5,'eight':8,'three':3,'twelve':12})
                df['bore'] = df['bore'].astype(float)
                df['stroke'] = df['stroke'].astype(float)
                df['horsepower'] = df['horsepower'].astype(float)
                df['peak_rpm'] = df['peak_rpm'].astype(float)
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
                #below code will create prediction vars if they aren't present in test set
                #if the below code is removed, test dataset will not work as get_dummies is used
                for i in pred_var:
                    if i not in df_dummies.columns:
                        df_dummies[i]=0
                df_dummies=df_dummies[pred_var]
                return df_dummies
        def fit(self, df_dummies, y=None, **fit_params):
                """This method is needed for writing preprocessing
                """  
                return self
        
def build_and_train():
    df = pd.read_csv('C:\\Users\\sagar\\Documents\\automobile.csv',names=['symboling','normalized_losses','make','fuel_type','aspiration','num_of_doors','body_style','drive_wheels','engine_location','wheel_base','length','width','height','curb_weight','engine_type',
'num_of_cylinders','engine_size','fuel_system','bore','stroke','compression_ratio','horsepower','peak_rpm','city_mpg','highway_mpg','price'])
   #hard coded SVR hyper parameters as obtained in full analysis.
    pipe = make_pipeline(preprocessing(),SVR(C=9.96, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
  kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False))
    #print(pipe)
    df.replace(to_replace='?',value=np.nan,inplace=True)
    df.dropna(subset=['num_of_doors','bore', 'stroke','horsepower', 'peak_rpm','price'],inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(df[df.columns.difference(['price'])],df['price'], test_size=0.25, random_state=42)
    pipe.fit(X_train, y_train)
    #though model was already worked out earlier, this is to save it
    return(pipe)
    

if __name__ == '__main__':
    model = build_and_train()
    #custom class preprocessing can be serialized with dill. Else, while
    #executing, it will throw an error saying preprocessing not found in __main__
    with open('C:/Users/sagar/Desktop/automobile/model.pkl', 'wb') as file:
        pickle.dump(model, file)
    print('dumped model!')