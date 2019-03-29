
# coding: utf-8

# In[13]:


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
import lightgbm as lgb
import catboost as cb
from sklearn.metrics import explained_variance_score,median_absolute_error
from sklearn.model_selection import cross_val_score
from IPython.display import Image
from sklearn.externals.six import StringIO
import pydotplus


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
rcParams['axes.titlepad'] = 20


# In[3]:


# Data Clensing
df = pd.read_csv('automobile.csv',names=['symboling','normalized_losses','make','fuel_type','aspiration','num_of_doors','body_style','drive_wheels','engine_location','wheel_base','length','width','height','curb_weight','engine_type',
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


# In[4]:


fig, ax = plt.subplots(nrows=5,ncols=3)
fig.set_figheight(25)
fig.set_figwidth(15)
ax = ax.reshape(-1)
for i in range(0,len(num_dtypes)):
    sns.regplot(x=num_dtypes[i],y='price',data=df,ax=ax[i],robust=True)
    fig.subplots_adjust(hspace=0.25)
    fig.subplots_adjust(wspace=0.25)


# In[5]:


fig, ax = plt.subplots(nrows=3,ncols=3)
fig.set_figheight(18)
fig.set_figwidth(15)
ax = ax.reshape(-1)
for i in range(0,len(object_dtypes)):
    sns.boxplot(x=object_dtypes[i],y='price',data=df,ax=ax[i])
    ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation = 90)
    fig.subplots_adjust(hspace=0.5)
    fig.subplots_adjust(wspace=0.25)


# In[6]:


fig, ax = plt.subplots(1,1)
stats.probplot(df['price'],plot=ax)
plt.show()


# In[7]:


for i in num_dtypes :
    df[i] = np.log1p(df[i])


# In[8]:


fig, ax = plt.subplots(1,1)
stats.probplot(df['price'],plot=ax)
plt.show()


# In[9]:


scaler = StandardScaler()
for j in num_dtypes :
    df[j] = scaler.fit_transform(df[[j]])


# In[10]:


df_dummies = pd.get_dummies(df,columns=['symboling','make','fuel_type','aspiration','body_style','drive_wheels','engine_type','fuel_system','engine_location'],drop_first=True)
df_dummies['Car_price'] = df_dummies['price']
df_dummies.drop(['price'],axis=1,inplace=True)


# In[11]:


rs = 10
df_dummies = df_dummies.sample(frac=1).reset_index(drop=True)
X = df_dummies.iloc[:,:-1]
y = df_dummies.iloc[:,-1]


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=rs)


# In[15]:


print(X_train.shape)
print(X_test.shape)


# In[16]:


model = sm.OLS(y_train,sm.add_constant(X_train))
results = model.fit()
print(results.summary())


# In[17]:


s_residuals = pd.Series(results.resid_pearson, name="S. Residuals")
fitted_values = pd.Series(results.fittedvalues, name="Fitted Values")
sns.regplot(fitted_values, s_residuals,  fit_reg=False)
plt.show()


# In[18]:


res = pd.DataFrame(s_residuals)
drop_res = res[abs(res['S. Residuals'])>2].index.values


# In[19]:


drop_res


# In[20]:


res = pd.DataFrame(s_residuals)
drop_res = res[abs(res['S. Residuals'])>2].index.values
X_train.reset_index(inplace=True)
X_train.drop('index',axis=1,inplace=True)
y_train.reset_index(inplace=True, drop=True)
X_train.drop(drop_res,inplace=True)
y_train.drop(drop_res,inplace=True)


# In[21]:


print(X_train.shape)
print(X_test.shape)


# In[22]:


model = sm.OLS(list(y_train),sm.add_constant(X_train))
results = model.fit()
print(results.summary())


# In[23]:


s_residuals = pd.Series(results.resid_pearson, name="S. Residuals")
leverage = pd.Series(OLSInfluence(results).influence, name = "Leverage")
sns.regplot(leverage, s_residuals,  fit_reg=False)
plt.show()


# In[24]:


l = pd.DataFrame(leverage)
drop_l = l[abs(l['Leverage'])>((X_train.shape[1])+1)/(X_train.shape[0])].index.values


# In[25]:


drop_l


# In[26]:


X_train.reset_index(inplace=True)
X_train.drop('index',axis=1,inplace=True)
y_train.reset_index(inplace=True, drop=True)
X_train.drop(drop_l,inplace=True)
y_train.drop(drop_l,inplace=True)


# In[27]:


print(X_train.shape)
print(X_test.shape)


# In[28]:


vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif["features"] = X_train.columns
vif


# In[29]:


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


# In[30]:


vif


# In[31]:


model = sm.OLS(y_train,sm.add_constant(X_train))
results = model.fit()
print(results.summary())


# In[32]:


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


# In[33]:


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


# In[34]:


master_columns = X_train.columns.values


# In[35]:


def add_interactions(df) :
    combos = list(combinations(list(df.columns),2))
    colnames = list(df.columns) + ['_'.join(x) for x in combos]
    poly = PolynomialFeatures(interaction_only=True, include_bias= False)
    df = poly.fit_transform(df)
    df = pd.DataFrame(df)
    df.columns = colnames
    return df


# In[36]:


X_test = X_test[X_train.columns]
X_train1 = add_interactions(X_train)
noint_indices = [i for i,x in enumerate(list((X_train1==0).all())) if x]
X_train1 = X_train1.drop(X_train1.columns[noint_indices],axis=1)
X_test1 =  add_interactions(X_test)
X_test1 = X_test1[X_train1.columns]
model = sm.OLS(list(y_train),sm.add_constant(X_train1))
results = model.fit()
print(results.summary())


# In[37]:


regression = LinearRegression()
regression.fit(X_train1,y_train)
pred = regression.predict(X_test1[X_train1.columns])
Predictions = pd.DataFrame(np.array([y_test.values,pred]).T,columns=['Original','Predicted'])
print(regression.score(X_test1[X_train1.columns],y_test))
sns.regplot(x='Original',y='Predicted',data=Predictions)
plt.show()


# In[38]:


result1 = stepwise_selection(X_train1, list(y_train))
for i in master_columns :
    if i not in (result1) :
        result1.append(i)
X_train1 = X_train1[result1]
model = sm.OLS(list(y_train),sm.add_constant(X_train1))
results = model.fit()
print(results.summary())


# In[39]:


accuracies = {}
MAE = {}


# ### Linear regression :

# In[40]:


def linear_reg(X_train1,y_train,X_test1,y_test,rs) :     
    regression = LinearRegression()
    regression.fit(X_train1,y_train)
    scores = cross_val_score(regression, X_train1, y_train,cv=10)
    print('Cross Validation scores: '+str(scores))
    print('Training Accuracy: '+str(scores.mean()))
    pred = regression.predict(X_test1[X_train1.columns])
    accuracies['LinearRegression']= explained_variance_score(y_test, pred)
    MAE['LinearRegression'] = median_absolute_error(y_test, pred)
    Predictions = pd.DataFrame(np.array([y_test.values,pred]).T,columns=['Original','Predicted'])
    print('Testing Accuracy: '+str(explained_variance_score(y_test, pred)))
    sns.regplot(x='Original',y='Predicted',data=Predictions)
    plt.show()


# In[41]:


linear_reg(X_train1,y_train,X_test1,y_test,rs)


# ### Lasso regression :

# In[42]:


def lasso_reg(X_train1,y_train,X_test1,y_test,rs) :    
    regression = LassoCV(cv=10,random_state=rs)
    regression.fit(X_train1,y_train)
    scores = cross_val_score(regression, X_train1, y_train,cv=10)
    print('Cross Validation scores: '+str(scores))
    print('Training Accuracy: '+str(scores.mean()))
    pred = regression.predict(X_test1[X_train1.columns])
    accuracies['Lasso']= explained_variance_score(y_test, pred)
    MAE['Lasso'] = median_absolute_error(y_test, pred)
    Predictions = pd.DataFrame(np.array([y_test.values,pred]).T,columns=['Original','Predicted'])
    print('Testing Accuracy: '+str(explained_variance_score(y_test, pred)))
    sns.regplot(x='Original',y='Predicted',data=Predictions)
    plt.show()


# In[43]:


lasso_reg(X_train1,y_train,X_test1,y_test,rs)


# ### Ridge regression :

# In[44]:


def ridge_reg(X_train1,y_train,X_test1,y_test,rs) :  
    regression = RidgeCV(cv=10)
    regression.fit(X_train1,y_train)
    scores = cross_val_score(regression, X_train1, y_train,cv=10)
    print('Cross Validation scores: '+str(scores))
    print('Training Accuracy: '+str(scores.mean()))
    pred = regression.predict(X_test1[X_train1.columns])
    accuracies['Ridge']= explained_variance_score(y_test, pred)
    MAE['Ridge'] = median_absolute_error(y_test, pred)
    Predictions = pd.DataFrame(np.array([y_test.values,pred]).T,columns=['Original','Predicted'])
    print('Testing Accuracy: '+str(explained_variance_score(y_test, pred)))
    sns.regplot(x='Original',y='Predicted',data=Predictions)
    plt.show()


# In[45]:


ridge_reg(X_train1,y_train,X_test1,y_test,rs)


# ### Support Vector regression :

# In[60]:


def SVR_reg(X_train1,y_train,X_test1,y_test,rs) :    
    param_grid = {'C':np.arange(0.01,10,0.05)}
    model = Gr(SVR(),param_grid, cv=10)
    model.fit(X_train1,y_train)
    model2 = model.best_estimator_
    scores = cross_val_score(model2, X_train1, y_train,cv=10)
    print('Cross Validation scores: '+str(scores))
    print('Training Accuracy: '+str(scores.mean()))
    pred = model2.predict(X_test1[X_train1.columns])
    accuracies['SVR']= explained_variance_score(y_test, pred)
    MAE['SVR'] = median_absolute_error(y_test, pred)
    Predictions = pd.DataFrame(np.array([y_test.values,pred]).T,columns=['Original','Predicted'])
    print('Testing Accuracy: '+str(explained_variance_score(y_test, pred)))
    sns.regplot(x='Original',y='Predicted',data=Predictions)
    plt.show()


# In[59]:


SVR_reg(X_train1,y_train,X_test1,y_test,rs)


# ### KNN regression :

# In[46]:


def KNN_reg(X_train1,y_train,X_test1,y_test,rs) : 
    param_grid = {'n_neighbors':np.arange(3,20)}
    model = GridSearchCV(KNeighborsRegressor(),param_grid, cv=10)
    model.fit(X_train1,y_train)
    model2 = model.best_estimator_
    scores = cross_val_score(model2, X_train1, y_train,cv=10)
    print('Cross Validation scores: '+str(scores))
    print('Training Accuracy: '+str(scores.mean()))
    pred = model2.predict(X_test1[X_train1.columns])
    accuracies['KNeighborsRegressor']= explained_variance_score(y_test, pred)
    MAE['KNeighborsRegressor'] = median_absolute_error(y_test, pred)
    Predictions = pd.DataFrame(np.array([y_test.values,pred]).T,columns=['Original','Predicted'])
    print('Testing Accuracy: '+str(explained_variance_score(y_test, pred)))
    sns.regplot(x='Original',y='Predicted',data=Predictions)
    plt.show()


# In[47]:


KNN_reg(X_train1,y_train,X_test1,y_test,rs)


# ### Decision Tree regression :

# In[48]:


def decision_tree_reg(X_train1,y_train,X_test1,y_test,rs) :
    param_grid1 = {'max_depth':np.arange(3,X_train1.shape[1])}
    param_grid2 = {'min_samples_split':np.linspace(0.1, 1.0, 10, endpoint=True)}
    param_grid3 = {'min_samples_leaf':np.linspace(0.1, 0.5, 5, endpoint=True)}
    param_grid4 = {'max_features':list(range(1,X_train1.shape[1]))}
    param_grid5 = {'max_leaf_nodes': np.arange(2, 20)}
    model1 = GridSearchCV(DecisionTreeRegressor(random_state=rs),param_grid1, cv=10)
    model1.fit(X_train1,y_train)
    model2 = GridSearchCV(model1.best_estimator_,param_grid2, cv=10)
    model2.fit(X_train1,y_train)
    model3 = GridSearchCV(model2.best_estimator_,param_grid3, cv=10)
    model3.fit(X_train1,y_train)
    model4 = GridSearchCV(model3.best_estimator_,param_grid4, cv=10)
    model4.fit(X_train1,y_train)
    model5 = GridSearchCV(model4.best_estimator_,param_grid5, cv=10)
    model5.fit(X_train1,y_train)
    model6 = model5.best_estimator_
    dot_data = StringIO()
    export_graphviz(model6,out_file=dot_data)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    print(model6)
    scores = cross_val_score(model6, X_train1, y_train,cv=10)
    print('Cross Validation scores: '+str(scores))
    print('Training Accuracy: '+str(scores.mean()))
    pred = model6.predict(X_test1[X_train1.columns])
    accuracies['DecisionTreeRegressor']= explained_variance_score(y_test, pred)
    MAE['DecisionTreeRegressor'] = median_absolute_error(y_test, pred)
    Predictions = pd.DataFrame(np.array([y_test.values,pred]).T,columns=['Original','Predicted'])
    print('Testing Accuracy: '+str(explained_variance_score(y_test, pred)))
    sns.regplot(x='Original',y='Predicted',data=Predictions)
    plt.show()
    return Image(graph.create_png())


# In[49]:


decision_tree_reg(X_train1,y_train,X_test1,y_test,rs)


# ### RandomForest Regression :

# In[50]:


def random_forest_reg(X_train1,y_train,X_test1,y_test,rs) :
    param_grid1 = {'n_estimators':[100,250,500,750,1000]}
    param_grid2 = {'max_depth':np.arange(3,X_train1.shape[1])}
    param_grid3 = {'min_samples_split':np.linspace(0.1, 1.0, 10, endpoint=True)}
    param_grid4 = {'min_samples_leaf':np.linspace(0.1, 0.5, 5, endpoint=True)}
    param_grid5 = {'max_features':list(range(1,X_train1.shape[1]))}
    param_grid6 = {'max_leaf_nodes': np.arange(2, 20)}
    model1 = GridSearchCV(RandomForestRegressor(random_state=rs),param_grid1, cv=10)
    model1.fit(X_train1,y_train)
    model2 = GridSearchCV(model1.best_estimator_,param_grid2, cv=10)
    model2.fit(X_train1,y_train)
    model3 = GridSearchCV(model2.best_estimator_,param_grid3, cv=10)
    model3.fit(X_train1,y_train)
    model4 = GridSearchCV(model3.best_estimator_,param_grid4, cv=10)
    model4.fit(X_train1,y_train)
    model5 = GridSearchCV(model4.best_estimator_,param_grid5, cv=10)
    model5.fit(X_train1,y_train)
    model6 = GridSearchCV(model5.best_estimator_,param_grid6, cv=10)
    model6.fit(X_train1,y_train)
    model7 = model6.best_estimator_
    scores = cross_val_score(model7, X_train1, y_train,cv=10)
    print('Cross Validation scores: '+str(scores))
    print('Training Accuracy: '+str(scores.mean()))
    pred = model7.predict(X_test1[X_train1.columns])
    accuracies['RandomForestRegressor']= explained_variance_score(y_test, pred)
    MAE['RandomForestRegressor'] = median_absolute_error(y_test, pred)
    Predictions = pd.DataFrame(np.array([y_test.values,pred]).T,columns=['Original','Predicted'])
    print('Testing Accuracy: '+str(explained_variance_score(y_test, pred)))
    sns.regplot(x='Original',y='Predicted',data=Predictions)
    plt.show()


# In[51]:


random_forest_reg(X_train1,y_train,X_test1,y_test,rs)


# ### GradientBoosting Regression :

# In[52]:


def gradient_boosting_reg(X_train1,y_train,X_test1,y_test,rs) :
    param_grid1 = {'learning_rate':[0.15,0.16,0.17,0.18,0.19,0.2]}
    param_grid2 = {'alpha':np.arange(0.1,0.9,0.1)}
    param_grid3 = {'n_estimators':[100,250,500,750,1000]}
    param_grid4 = {'max_depth':np.arange(3,X_train1.shape[1])}
    param_grid5 = {'min_samples_split':np.linspace(0.1, 1.0, 10, endpoint=True)}
    param_grid6 = {'min_samples_leaf':np.linspace(0.1, 0.5, 5, endpoint=True)}
    param_grid7 = {'max_features':list(range(1,X_train1.shape[1]))}
    param_grid8 = {'max_leaf_nodes': np.arange(2, 20)}
    model1 = GridSearchCV(GradientBoostingRegressor(random_state=rs,),param_grid1, cv=10)
    model1.fit(X_train1,y_train)
    model2 = GridSearchCV(model1.best_estimator_,param_grid2, cv=10)
    model2.fit(X_train1,y_train)
    model3 = GridSearchCV(model2.best_estimator_,param_grid3, cv=10)
    model3.fit(X_train1,y_train)
    model4 = GridSearchCV(model3.best_estimator_,param_grid4, cv=10)
    model4.fit(X_train1,y_train)
    model5 = GridSearchCV(model4.best_estimator_,param_grid5, cv=10)
    model5.fit(X_train1,y_train)
    model6 = GridSearchCV(model5.best_estimator_,param_grid6, cv=10)
    model6.fit(X_train1,y_train)
    model7 = GridSearchCV(model6.best_estimator_,param_grid7, cv=10)
    model7.fit(X_train1,y_train)
    model8 = GridSearchCV(model7.best_estimator_,param_grid8, cv=10)
    model8.fit(X_train1,y_train)
    model9 = model8.best_estimator_
    scores = cross_val_score(model9, X_train1, y_train,cv=10)
    print('Cross Validation scores: '+str(scores))
    print('Training Accuracy: '+str(scores.mean()))
    pred = model9.predict(X_test1[X_train1.columns])
    accuracies['GradientBoostingRegressor']= explained_variance_score(y_test, pred)
    MAE['GradientBoostingRegressor'] = median_absolute_error(y_test, pred)
    Predictions = pd.DataFrame(np.array([y_test.values,pred]).T,columns=['Original','Predicted'])
    print('Testing Accuracy: '+str(explained_variance_score(y_test, pred)))
    sns.regplot(x='Original',y='Predicted',data=Predictions)
    plt.show()


# In[53]:


gradient_boosting_reg(X_train1,y_train,X_test1,y_test,rs)


# ### AdaBoost Regressor :

# In[54]:


def ada_boost_reg(X_train1,y_train,X_test1,y_test,rs) :
    param_grid1 = {'n_estimators':[100,250,500,750,1000]}
    param_grid2 = {'learning_rate':[0.15,0.16,0.17,0.18,0.19,0.2]}
    model1 = GridSearchCV(AdaBoostRegressor(random_state=rs),param_grid1, cv=10)
    model1.fit(X_train1,y_train)
    model2 = GridSearchCV(model1.best_estimator_,param_grid2, cv=10)
    model2.fit(X_train1,y_train)
    model3 = model2.best_estimator_
    scores = cross_val_score(model3, X_train1, y_train,cv=10)
    print('Cross Validation scores: '+str(scores))
    print('Training Accuracy: '+str(scores.mean()))
    pred = model3.predict(X_test1[X_train1.columns])
    accuracies['AdaBoostRegressor']= explained_variance_score(y_test, pred)
    MAE['AdaBoostRegressor'] = median_absolute_error(y_test, pred)
    Predictions = pd.DataFrame(np.array([y_test.values,pred]).T,columns=['Original','Predicted'])
    print('Testing Accuracy: '+str(explained_variance_score(y_test, pred)))
    sns.regplot(x='Original',y='Predicted',data=Predictions)
    plt.show()


# In[55]:


ada_boost_reg(X_train1,y_train,X_test1,y_test,rs)


# ### XGBoost Regressor :

# In[56]:


def XG_boost_reg(X_train1,y_train,X_test1,y_test,rs) :
    param_grid1 = {'n_estimators':[100,250,500,750,1000]}
    param_grid2 = {'learning_rate':[0.05,0.1,0.15,0.2,0.25,0.5,0.75,1]}
    param_grid3 = {'booster':['gbtree','gblinear','dart']}
    param_grid4 = {'base_score':np.arange(0.1,1,0.1)}
    param_grid5 = {'reg_alpha':np.arange(0.1,1,0.1)}
    param_grid6 = {'reg_lambda':np.arange(0.1,2,0.1)}
    param_grid7 = {'gamma':np.arange(0.1,2,0.1)}
    param_grid8 = {'max_depth':np.arange(3,X_train1.shape[1])}
    model1 = GridSearchCV(xgb.XGBRegressor(random_state=rs),param_grid1, cv=10)
    model1.fit(X_train1,y_train)
    model2 = GridSearchCV(model1.best_estimator_,param_grid2, cv=10)
    model2.fit(X_train1,y_train)
    model3 = GridSearchCV(model2.best_estimator_,param_grid3, cv=10)
    model3.fit(X_train1,y_train)
    model4 = GridSearchCV(model3.best_estimator_,param_grid4, cv=10)
    model4.fit(X_train1,y_train)
    model5 = GridSearchCV(model4.best_estimator_,param_grid5, cv=10)
    model5.fit(X_train1,y_train)
    model6 = GridSearchCV(model5.best_estimator_,param_grid6, cv=10)
    model6.fit(X_train1,y_train)
    model7 = GridSearchCV(model6.best_estimator_,param_grid7, cv=10)
    model7.fit(X_train1,y_train)
    model8 = GridSearchCV(model7.best_estimator_,param_grid8, cv=10)
    model8.fit(X_train1,y_train)
    model9 = model8.best_estimator_
    scores = cross_val_score(model9, X_train1, y_train,cv=10)
    print('Cross Validation scores: '+str(scores))
    print('Training Accuracy: '+str(scores.mean()))
    pred = model9.predict(X_test1[X_train1.columns])
    accuracies['XGBRegressor']= explained_variance_score(y_test, pred)
    MAE['XGBRegressor'] = median_absolute_error(y_test, pred)
    Predictions = pd.DataFrame(np.array([y_test.values,pred]).T,columns=['Original','Predicted'])
    print('Testing Accuracy: '+str(explained_variance_score(y_test, pred)))
    sns.regplot(x='Original',y='Predicted',data=Predictions)
    plt.show()


# In[57]:


XG_boost_reg(X_train1,y_train,X_test1,y_test,rs)


# ### LGBM Regressor :

# In[58]:


def lgbm_reg(X_train1,y_train,X_test1,y_test,rs) :
   param_grid1 = {'n_estimators':[100,250,500,750,1000]}
   param_grid2 = {'max_depth': np.arange(3,X_train1.shape[1])}
   param_grid3 = {'num_leaves':np.arange(5,100,5)}
   param_grid4 = {'learning_rate':[0.15,0.16,0.17,0.18,0.19,0.2]}
   param_grid5 = {'boosting_type':['gbdt','dart','goss']}
   param_grid6 = {'reg_alpha':np.arange(0.1,1,0.1)}
   param_grid7 = {'reg_lambda':np.arange(0.1,1,0.1)}
   param_grid8 = {'subsample_for_bin':[100,250,500,750,1000,5000,10000,50000,100000,200000]}
   model1 = GridSearchCV(lgb.LGBMRegressor(random_state=rs,),param_grid1, cv=10)
   model1.fit(X_train1,y_train)
   model2 = GridSearchCV(model1.best_estimator_,param_grid2, cv=10)
   model2.fit(X_train1,y_train)
   model3 = GridSearchCV(model2.best_estimator_,param_grid3, cv=10)
   model3.fit(X_train1,y_train)
   model4 = GridSearchCV(model3.best_estimator_,param_grid4, cv=10)
   model4.fit(X_train1,y_train)
   model5 = GridSearchCV(model4.best_estimator_,param_grid5, cv=10)
   model5.fit(X_train1,y_train)
   model6 = GridSearchCV(model5.best_estimator_,param_grid6, cv=10)
   model6.fit(X_train1,y_train)
   model7 = GridSearchCV(model6.best_estimator_,param_grid7, cv=10)
   model7.fit(X_train1,y_train)
   model8 = GridSearchCV(model7.best_estimator_,param_grid7, cv=10)
   model8.fit(X_train1,y_train)
   model9 = model8.best_estimator_
   scores = cross_val_score(model9, X_train1, y_train,cv=10)
   print('Cross Validation scores: '+str(scores))
   print('Training Accuracy: '+str(scores.mean()))
   pred = model9.predict(X_test1[X_train1.columns])
   accuracies['LGBMRegressor']= explained_variance_score(y_test, pred)
   MAE['LGBMRegressor'] = median_absolute_error(y_test, pred)
   Predictions = pd.DataFrame(np.array([y_test.values,pred]).T,columns=['Original','Predicted'])
   print('Testing Accuracy: '+str(explained_variance_score(y_test, pred)))
   sns.regplot(x='Original',y='Predicted',data=Predictions)
   plt.show()


# In[59]:


lgbm_reg(X_train1,y_train,X_test1,y_test,rs)


# ### CatBoost Regressor :

# In[60]:


def cat_boost_reg(X_train1,y_train,X_test1,y_test,rs) :
    param_grid1 = {'learning_rate':[0.15,0.16,0.17,0.18,0.19,0.2]}
    param_grid2 = {'depth':[3,4,5,7,9,None]}
    model1 = GridSearchCV(cb.CatBoostRegressor(random_state=rs,iterations=(X_train1.shape[0])/3),param_grid1, cv=10)
    model1.fit(X_train1,y_train)
    model2 = GridSearchCV(model1.best_estimator_,param_grid2, cv=10)
    model2.fit(X_train1,y_train)
    model3 = model2.best_estimator_
    scores = cross_val_score(model3, X_train1, y_train,cv=10)
    print('Cross Validation scores: '+str(scores))
    print('Training Accuracy: '+str(scores.mean()))
    pred = model3.predict(X_test1[X_train1.columns])
    accuracies['CatBoostRegressor ']= explained_variance_score(y_test, pred)
    MAE['CatBoostRegressor '] = median_absolute_error(y_test, pred)
    Predictions = pd.DataFrame(np.array([y_test.values,pred]).T,columns=['Original','Predicted'])
    print('Testing Accuracy: '+str(explained_variance_score(y_test, pred)))
    sns.regplot(x='Original',y='Predicted',data=Predictions)
    plt.show()


# In[61]:


cat_boost_reg(X_train1,y_train,X_test1,y_test,rs)


# In[62]:


scores= pd.DataFrame.from_dict(accuracies,orient='index').reset_index()
scores.columns=['Algorithm','Score']
sns.pointplot(x='Algorithm',y='Score',data=scores)
plt.xticks(rotation=90)
plt.show()


# In[63]:


error = pd.DataFrame.from_dict(MAE,orient='index').reset_index()
error.columns=['Algorithm','MAE']
sns.pointplot(x='Algorithm',y='MAE',data=error)
plt.xticks(rotation=90)
plt.show()

