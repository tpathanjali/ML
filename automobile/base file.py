# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 20:07:28 2019

@author: sagar
"""


# coding: utf-8

# In[4]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from matplotlib import rcParams
from scipy import stats
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LassoCV,RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor,export_graphviz
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from itertools import combinations
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
#import lightgbm as lgb
import catboost as cb
from sklearn.metrics import explained_variance_score,median_absolute_error
from sklearn.model_selection import cross_val_score
from IPython.display import Image
from sklearn.externals.six import StringIO
#import pydotplus


# In[6]:


# Data Clensing
df = pd.read_csv('C:\\Users\\sagar\\Documents\\automobile.csv',names=['symboling','normalized_losses','make','fuel_type','aspiration','num_of_doors','body_style','drive_wheels','engine_location','wheel_base','length','width','height','curb_weight','engine_type',
'num_of_cylinders','engine_size','fuel_system','bore','stroke','compression_ratio','horsepower','peak_rpm','city_mpg','highway_mpg','price'])
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


# In[7]:


for i in num_dtypes :
    df[i] = np.log1p(df[i])


# In[8]:


scaler = StandardScaler()
for j in num_dtypes :
    df[j] = scaler.fit_transform(df[[j]])


# In[9]:


df_dummies = pd.get_dummies(df,columns=['symboling','make','fuel_type','aspiration','body_style','drive_wheels','engine_type','fuel_system','engine_location'],drop_first=True)
df_dummies['Car_price'] = df_dummies['price']
df_dummies.drop(['price'],axis=1,inplace=True)
X = df_dummies.iloc[:,:-1]
y = df_dummies.iloc[:,-1]


# In[10]:


rs = 10


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=rs)


# In[13]:


print(X_train.shape)
print(y_train.shape)


# In[14]:


model = sm.OLS(y_train,sm.add_constant(X_train))
results = model.fit()
print(results.summary())


# In[15]:


s_residuals = pd.Series(results.resid_pearson, name="S. Residuals")
fitted_values = pd.Series(results.fittedvalues, name="Fitted Values")
sns.regplot(fitted_values, s_residuals,  fit_reg=False)
plt.show()


# In[16]:


res = pd.DataFrame(s_residuals)
drop_res = res[abs(res['S. Residuals'])>2].index.values


# In[17]:


drop_res


# In[18]:


res = pd.DataFrame(s_residuals)
drop_res = res[abs(res['S. Residuals'])>2].index.values
X_train.reset_index(inplace=True)
X_train.drop('index',axis=1,inplace=True)
y_train.reset_index(inplace=True, drop=True)
X_train.drop(drop_res,inplace=True)
y_train.drop(drop_res,inplace=True)


# In[19]:


print(X_train.shape)
print(X_test.shape)


# In[20]:


model = sm.OLS(list(y_train),sm.add_constant(X_train))
results = model.fit()
print(results.summary())


# In[21]:


s_residuals = pd.Series(results.resid_pearson, name="S. Residuals")
leverage = pd.Series(OLSInfluence(results).influence, name = "Leverage")
sns.regplot(leverage, s_residuals,  fit_reg=False)
plt.show()


# In[22]:


l = pd.DataFrame(leverage)
drop_l = l[abs(l['Leverage'])>((X_train.shape[1])+1)/(X_train.shape[0])].index.values


# In[23]:


drop_l


# In[24]:


X_train.reset_index(inplace=True)
X_train.drop('index',axis=1,inplace=True)
y_train.reset_index(inplace=True, drop=True)
X_train.drop(drop_l,inplace=True)
y_train.drop(drop_l,inplace=True)


# In[25]:


print(X_train.shape)
print(X_test.shape)


# In[26]:


vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif["features"] = X_train.columns


# In[27]:


c = True
while c :
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    vif["features"] = X_train.columns
    v = vif[vif.index==vif['VIF Factor'].idxmax()]['VIF Factor'].values[0]
    if v > 10 :
        X_train.drop(vif[vif.index==vif['VIF Factor'].idxmax()]['features'].values,inplace=True,axis=1)
    else :
        c = False


# In[29]:


model = sm.OLS(y_train,sm.add_constant(X_train))
results = model.fit()
print(results.summary())


# In[30]:


def stepwise_selection(X, y, 
                       initial_list=[], 
                       threshold_in=0.05, 
                       threshold_out = 0.05, 
                       verbose=True):
    included = list(initial_list)
    while True:
        changed=False
        # forward step
        excluded = list(set(X.columns)-set(included))
        new_pval = pd.Series(index=excluded)
        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
            new_pval[new_column] = model.pvalues[new_column]
        best_pval = new_pval.min()
        if best_pval < threshold_in:
            best_feature = new_pval.argmin()
            included.append(best_feature)
            changed=True
        model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
        # use all coefs except intercept
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max() # null if pvalues is empty
        if worst_pval > threshold_out:
            changed=True
            worst_feature = pvalues.argmax()
            included.remove(worst_feature)
        if not changed:
            break
    return included


# In[31]:


result = stepwise_selection(X_train, y_train)
X_train = X_train[result]
model = sm.OLS(y_train,sm.add_constant(X_train))
results = model.fit()
print(results.summary())
regression = LinearRegression()
regression.fit(X_train,y_train)
pred = regression.predict(X_test[X_train.columns])
Predictions = pd.DataFrame(np.array([y_test.values,pred]).T,columns=['Original','Predicted'])
print(regression.score(X_test[X_train.columns],y_test))
sns.regplot(x='Original',y='Predicted',data=Predictions)
plt.show()


# In[32]:


master_columns = X_train.columns.values


# In[33]:


def add_interactions(df) :
    combos = list(combinations(list(df.columns),2))
    colnames = list(df.columns) + ['_'.join(x) for x in combos]
    poly = PolynomialFeatures(interaction_only=True, include_bias= False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames
    return df


# In[34]:


X_test = X_test[X_train.columns]
X_train1 = add_interactions(X_train)
noint_indices = [i for i,x in enumerate(list((X_train1==0).all())) if x]
X_train1 = X_train1.drop(X_train1.columns[noint_indices],axis=1)
X_test1 =  add_interactions(X_test)
X_test1 = X_test1[X_train1.columns]
model = sm.OLS(list(y_train),sm.add_constant(X_train1))
results = model.fit()
print(results.summary())


# In[35]:


regression = LinearRegression()
regression.fit(X_train1,y_train)
pred = regression.predict(X_test1[X_train1.columns])
Predictions = pd.DataFrame(np.array([y_test.values,pred]).T,columns=['Original','Predicted'])
print(regression.score(X_test1[X_train1.columns],y_test))
#sns.regplot(x='Original',y='Predicted',data=Predictions)
#plt.show()


# In[36]:


result1 = stepwise_selection(X_train1, list(y_train))
for i in master_columns :
    if i not in (result1) :
        result1.append(i)
X_train1 = X_train1[result1]
model = sm.OLS(list(y_train),sm.add_constant(X_train1))
results = model.fit()
print(results.summary())

from sklearn.externals import joblib
def SVR_reg(X_train,y_train,X_test,y_test,rs) :    
    param_grid = {'C':np.arange(0.01,10,0.05)}
    model = GridSearchCV(SVR(),param_grid, cv=10)
    model.fit(X_train,y_train)
    model2 = model.best_estimator_
    print(model2)
    scores = cross_val_score(model2, X_train, y_train,cv=10)
    pred = model2.predict(X_test[X_train.columns])
    model_columns = list(X_test.columns)
    joblib.dump(model2, 'model2.pkl')
    print('dumped model!')
    joblib.dump(model_columns, 'automobile_columns.pkl')
    print('dumped columns!!')
    plt.show()


# In[71]:


SVR_reg(X_train,y_train,X_test,y_test,rs)



