from My_Regression_Class import MyRegressionClass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import VotingClassifier
from yellowbrick.classifier import ROCAUC, PrecisionRecallCurve, ClassPredictionError, DiscriminationThreshold
from yellowbrick.model_selection import ValidationCurve, LearningCurve
import time
from sklearn.model_selection import StratifiedKFold
import matplotlib

boston = load_boston()
boston.items()
boston['data']
columns = boston['feature_names']

df = pd.DataFrame(boston['data'], columns=columns)
df['target']=boston['target']

X = df.drop('target', axis=1)
y = df['target']

regression = MyRegressionClass(test_size=0.2)
X_train,X_test,y_train,y_test = regression.TrainTestSet(X,y)
results,y_pred,regression_model = regression.AllEstimatorsTito(X_train,X_test,y_train,y_test)
regression.ResidualPlotTito(regression_model,X_train,X_test,y_train,y_test)
regression.PredictionErrorTito(regression_model,X_train,X_test,y_train,y_test)
regression.AlphaSelectionTito(regression_model,X_train,y_train)
regression.CookDistanceTito(regression_model,X_train,y_train)



















lr_regressor = LinearRegression(fit_intercept=True)
rd_regressor = Ridge(alpha=50)
la_regressor = Lasso(alpha=50)
lr_poly_regressor= LinearRegression(fit_intercept=True)
svr_regressor = SVR(kernel = 'rbf')
dt_regressor = DecisionTreeRegressor(random_state=0)
rf_regressor = RandomForestRegressor(n_estimators=300, random_state=0)
ad_regressor = AdaBoostRegressor()
gb_regressor = GradientBoostingRegressor()
xgb_regressor = XGBRegressor()

model = [('Logistic Regression', lr_regressor),
         ('Ridge Regression', rd_regressor),
         ('Lasso Regression', la_regressor),
         ('Polynomial Regression', lr_poly_regressor),
         ('Support Vector RBF', svr_regressor),
         ('Decision Tree Regression', dt_regressor),
         ('Random Forest Regression', rf_regressor),
         ('Ada Boosting', ad_regressor),
         ('Gradient Boosting', gb_regressor),
         ('Xg Boosting', xgb_regressor)]
results = pd.DataFrame()
y_predict = []
        
for models,estimator in model:
    if models == 'Polynomial Regression':
        k = X_test.shape[1]
        n = len(X_test)
        poly_reg = PolynomialFeatures(degree = 2)
        X_poly = poly_reg.fit_transform(X_train)
        estimator.fit(X_poly, y_train)
        y_pred = estimator.predict(poly_reg.fit_transform(X_test))
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r2 = metrics.r2_score(y_test, y_pred)
        adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
        MAPE = np.mean(np.abs((y_test-y_pred) / y_test))*100
        model_results = pd.DataFrame([[models, mae, mse, rmse, r2,adj_r2]],
                                     columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score'])
        results = results.append(model_results, ignore_index = True)
        y_predict.append((models,list(y_pred)))
    
    if models == 'Support Vector RBF':   
        from sklearn.preprocessing import StandardScaler
        sc_x = StandardScaler()
        sc_y = StandardScaler()
        X_train = pd.DataFrame(sc_x.fit_transform(X_train), columns=X_train.columns.values)
        X_test = pd.DataFrame(sc_x.transform(X_test), columns=X_train.columns.values)
        y_test = pd.DataFrame(sc_y.fit_transform(np.array(y_test.iloc[:]).reshape(len(y_test),1)), columns=[y_test.name])
        y_train = pd.DataFrame(sc_y.transform(np.array(y_train.iloc[:]).reshape(len(y_train),1)), columns=[y_train.name])
        k = X_test.shape[1]
        n = len(X_test)
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r2 = metrics.r2_score(y_test, y_pred)
        adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
        model_results = pd.DataFrame([[models, mae, mse, rmse, r2,adj_r2,]],
                                     columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score'])
        results = results.append(model_results, ignore_index = True)
    
    else:
        k = X_test.shape[1]
        n = len(X_test)
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        mae = metrics.mean_absolute_error(y_test, y_pred)
        mse = metrics.mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r2 = metrics.r2_score(y_test, y_pred)
        adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)
        model_results = pd.DataFrame([[models, mae, mse, rmse, r2,adj_r2]],
                                     columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score'])
        results = results.append(model_results, ignore_index = True)
        y_predict.append((models,list(y_pred)))

##Ensemble Voting regressor
from sklearn.ensemble import VotingRegressor
voting_regressor = VotingRegressor(estimators= [('lr', lr_regressor),
                                                ('rd', rd_regressor),
                                                ('la', la_regressor),
                                                ('lr_poly', lr_poly_regressor),
                                                ('svr', svr_regressor),
                                                ('dt', dt_regressor),
                                                ('rf', rf_regressor),
                                                ('ad', ad_regressor),
                                                ('gr', gb_regressor),
                                                ('xg', xgb_regressor)])

for clf in (lr_regressor,lr_poly_regressor,svr_regressor,dt_regressor,
            rf_regressor, ad_regressor,gb_regressor, xgb_regressor, voting_regressor):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, metrics.r2_score(y_test, y_pred))

# Predicting Test Set
y_pred = voting_regressor.predict(X_test)
from sklearn import metrics
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2) * (n-1)/(n-k-1)

model_results = pd.DataFrame([['Ensemble Voting', mae, mse, rmse, r2,adj_r2]],
               columns = ['Model', 'MAE', 'MSE', 'RMSE', 'R2 Score','Adj. R2 Score'])

results = results.append(model_results, ignore_index = True)  
y_predict.append(('Ensemble Voting',list(y_pred)))

#The Best Classifier
the_best_estimator = results.sort_values(by='Adj. R2 Score',ascending=False).head(5)
print('The best regressor is:')
print('{}'.format(the_best_estimator))

y_pred = pd.DataFrame(y_predict, columns=['Model', 'y_pred'])
y_pred = y_pred[y_pred['Model']==the_best_estimator.iloc[0,0]]['y_pred']
y_pred = list(y_pred)
y_pred[0]
            
model_df = pd.DataFrame(model, columns=['Model', 'Estimator'])
estimator = model_df[model_df['Model']==the_best_estimator.iloc[0,0]]['Estimator']
estimator = list(estimator)
estimator[0]














       