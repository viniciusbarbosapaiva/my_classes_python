from My_Classification_Class import MyClassificationClass
from sklearn.datasets import load_iris,load_breast_cancer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
import time

time.strftime("%H:%M:%S",time.gmtime(948.9148545265198))

cancer = load_breast_cancer()
cancer.items()
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df['target'] = cancer['target']

X = df.drop('target', axis=1)
y = df['target']

training = MyClassificationClass(stratify=y,binary=True)
X_train,X_test,y_train,y_test = training.TrainTestSet(X,y)
results,y_pred,proba,all_estimator = training.AllEstimatorsTito(X_train,X_test,y_train,y_test)
training.ConfusionMatrixTito(y_test,y_pred)
training.ROCTito(all_estimator,X_train,X_test,y_train,y_test)
training.PrecisionRecallTito(all_estimator,X_train,X_test,y_train,y_test)
training.ClassPredictionErrorTito(all_estimator,X_train,X_test,y_train,y_test)
training.DiscriminationThresholdrTito(all_estimator,X_train,X_test,y_train,y_test)
training.LearningCurveTito(all_estimator, X_train,y_train, 'accuracy')
training.ValidationCurveTito(all_estimator, X_train,y_train, 'gamma', 'accuracy')

################################################################################

lr_classifier = LogisticRegression(random_state = 0, penalty = 'l2')
kn_classifier = KNeighborsClassifier(n_neighbors=15, metric='minkowski', p= 2)
svm_linear_classifier = SVC(random_state = 0, kernel = 'linear', probability= True)
svm_rbf_classifier= SVC(random_state = 0, kernel = 'rbf', probability= True)
gb_classifier = GaussianNB()
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
rf_classifier = RandomForestClassifier(random_state = 0, n_estimators = 100, criterion = 'gini')
ad_classifier = AdaBoostClassifier()
gr_classifier = GradientBoostingClassifier()
xg_classifier = XGBClassifier()

model = [('Logistic Regression (Lasso)', lr_classifier),
         ('K-Nearest Neighbors (minkowski)', kn_classifier),
         ('SVM (Linear)', svm_linear_classifier),
         ('SVM (RBF)', svm_rbf_classifier),
         ('Naive Bayes (Gaussian)', gb_classifier),
         ('Decision Tree', dt_classifier),
         ('Random Forest Gini (n=100)', rf_classifier),
         ('Ada Boosting', ad_classifier),
         ('Gradient Boosting', gr_classifier),
         ('Xg Boosting', xg_classifier)]

results = pd.DataFrame()
proba = []
y_predict = []

for models, estimator in model:
    if models == 'K-Nearest Neighbors (minkowski)':
        error_rate= []
        for i in np.arange(1,40):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            error_rate.append(np.mean(pred_i != y_test))
        n_neighbors = pd.DataFrame({'K':np.arange(1,40),
                                    'Error_Rate':error_rate})
        n_neighbors = n_neighbors['Error_Rate'].argmin()
        print("The smallest error was with 'K' = {}".format(n_neighbors))
    
        estimator.fit(X_train, y_train)    
        y_pred = estimator.predict(X_test)
        probability = estimator.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred)
        model_results = pd.DataFrame([[models, acc, prec, rec, f1,roc]],
                             columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])    
        results = results.append(model_results, ignore_index = True)
        proba.append((models,probability))
        y_predict.append((models,list(y_pred)))
        
    else:
        estimator.fit(X_train, y_train)    
        y_pred = estimator.predict(X_test)
        probability = estimator.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred)
        model_results = pd.DataFrame([[models, acc, prec, rec, f1,roc]],
                             columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])    
        results = results.append(model_results, ignore_index = True)
        proba.append((models,probability))
        y_predict.append((models,list(y_pred)))
        print("Not 'K-Nearest Neighbors (minkowski)'")

voting_classifier = VotingClassifier(estimators= [('lr', lr_classifier),
                                                  ('kn', kn_classifier),
                                                  ('svc_linear', svm_linear_classifier),
                                                  ('svc_rbf', svm_rbf_classifier),
                                                  ('gb', gb_classifier),
                                                  ('dt', dt_classifier),
                                                  ('rf', rf_classifier),
                                                  ('ad', ad_classifier),
                                                  ('gr', gr_classifier),
                                                  ('xg', xg_classifier),],
voting='soft')

for clf in (lr_classifier,kn_classifier,svm_linear_classifier,svm_rbf_classifier,
            gb_classifier, dt_classifier,rf_classifier, ad_classifier, gr_classifier, xg_classifier,
            voting_classifier):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# Predicting Test Set
y_pred = voting_classifier.predict(X_test)
probability = estimator.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred)
model_results = pd.DataFrame([['Ensemble Voting', acc, prec, rec, f1, roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])

results = results.append(model_results, ignore_index = True)
proba.append(('Ensemble Voting',probability))
y_predict.append(('Ensemble Voting',list(y_pred)))

#The Best Classifier
the_best_estimator = results.sort_values(by='Accuracy',ascending=False).head(5)
print('The best classifier was:')
print('{}'.format(the_best_estimator))

y_pred = pd.DataFrame(y_predict, columns=['Model', 'y_pred'])
y_pred[y_pred['Model']==the_best_estimator.iloc[0,0]]['y_pred']
y_pred_final = y_pred[y_pred['Model']==the_best_estimator.iloc[0,0]]['y_pred']
y_pred_final = list(y_pred_final)
y_pred_final[0]

proba = pd.DataFrame(proba, columns=['Model', 'Proba'])
proba[proba['Model']==the_best_estimator.iloc[0,0]]['Proba']
proba_final = proba[proba['Model']==the_best_estimator.iloc[0,0]]['Proba']
proba_final = list(proba_final)
proba_final[0]


model_df = pd.DataFrame(model, columns=['Model', 'Estimator'])
model_df[model_df['Model']==the_best_estimator.iloc[0,0]]['Estimator']
model_df_final = model_df[model_df['Model']==the_best_estimator.iloc[0,0]]['Estimator']
model_df_final = list(model_df_final)
model_df_final[0]

training.ConfusionMatrixTito(y_test,y_pred_final[0])
training.ROCTito(model_df_final[0],X_train,X_test,y_train,y_test)

###################################################################################################
iris = load_iris()
iris.items()
df2 = pd.DataFrame(iris['data'],columns=iris['feature_names'])
df2['target'] = iris['target']

X = df2.drop('target', axis=1)
y = df2['target']

training = MyClassificationClass(stratify=y,binary=False)
X_train,X_test,y_train,y_test = training.TrainTestSet(X,y)
results,y_pred,proba,all_estimator = training.AllEstimatorsTito(X_train,X_test,y_train,y_test)
training.ConfusionMatrixTito(y_test,y_pred)
training.ROCTito(all_estimator,X_train,X_test,y_train,y_test)
training.PrecisionRecallTito(all_estimator,X_train,X_test,y_train,y_test)
training.ClassPredictionErrorTito(all_estimator,X_train,X_test,y_train,y_test)
training.DiscriminationThresholdrTito(all_estimator,X_train,X_test,y_train,y_test)
training.LearningCurveTito(all_estimator, X_train,y_train, 'accuracy')
training.ValidationCurveTito(all_estimator, X_train,y_train, 'C', 'accuracy')

###########################################################################################

lr_classifier = LogisticRegression(random_state = 0, penalty = 'l2')
kn_classifier = KNeighborsClassifier(n_neighbors=15, metric='minkowski', p= 2)
svm_linear_classifier = SVC(random_state = 0, kernel = 'linear', probability= True)
svm_rbf_classifier= SVC(random_state = 0, kernel = 'rbf', probability= True)
gb_classifier = GaussianNB()
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
rf_classifier = RandomForestClassifier(random_state = 0, n_estimators = 100, criterion = 'gini')
ad_classifier = AdaBoostClassifier()
gr_classifier = GradientBoostingClassifier()
xg_classifier = XGBClassifier()
model = [('Logistic Regression (Lasso)', lr_classifier),
         ('K-Nearest Neighbors (minkowski)', kn_classifier),
         ('SVM (Linear)', svm_linear_classifier),
         ('SVM (RBF)', svm_rbf_classifier),
         ('Naive Bayes (Gaussian)', gb_classifier),
         ('Decision Tree', dt_classifier),
         ('Random Forest Gini (n=100)', rf_classifier),
         ('Ada Boosting', ad_classifier),
         ('Gradient Boosting', gr_classifier),
         ('Xg Boosting', xg_classifier)]
results = pd.DataFrame()
proba = []
y_predict = []

for models, estimator in model:
    if models == 'K-Nearest Neighbors (minkowski)':
        error_rate= []
        for i in np.arange(1,40):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            error_rate.append(np.mean(pred_i != y_test))
        n_neighbors = pd.DataFrame({'K':np.arange(1,40),
                                    'Error_Rate':error_rate})
        n_neighbors = n_neighbors['Error_Rate'].argmin()
        print("The smallest error was with 'K' = {}".format(n_neighbors))
    
        estimator.fit(X_train, y_train)    
        y_pred = estimator.predict(X_test)
        probability = estimator.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='micro')
        rec = recall_score(y_test, y_pred, average='micro')
        f1 = f1_score(y_test, y_pred, average='macro')
        roc = roc_auc_score(y_test, probability, multi_class='ovr')
        model_results = pd.DataFrame([[models, acc, prec, rec, f1,roc]],
                                     columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])    
        results = results.append(model_results, ignore_index = True)
        proba.append((models,probability))
        y_predict.append((models,list(y_pred)))
    
    else:
        estimator.fit(X_train, y_train)    
        y_pred = estimator.predict(X_test)
        probability = estimator.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='micro')
        rec = recall_score(y_test, y_pred, average='micro')
        f1 = f1_score(y_test, y_pred, average='macro')
        roc = roc_auc_score(y_test, probability, multi_class='ovr')
        model_results = pd.DataFrame([[models, acc, prec, rec, f1,roc]],
                                     columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])    
        results = results.append(model_results, ignore_index = True)
        proba.append((models,probability))
        y_predict.append((models,list(y_pred)))
        print("Not 'K-Nearest Neighbors (minkowski)'")

voting_classifier = VotingClassifier(estimators= [('lr', lr_classifier),
                                                  ('kn', kn_classifier),
                                                  ('svc_linear', svm_linear_classifier),
                                                  ('svc_rbf', svm_rbf_classifier),
                                                  ('gb', gb_classifier),
                                                  ('dt', dt_classifier),
                                                  ('rf', rf_classifier),
                                                  ('ad', ad_classifier),
                                                  ('gr', gr_classifier),
                                                  ('xg', xg_classifier),], voting='soft')
            
for clf in (lr_classifier,kn_classifier,svm_linear_classifier,svm_rbf_classifier,
            gb_classifier, dt_classifier,rf_classifier, ad_classifier, gr_classifier, xg_classifier, voting_classifier):
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

# Predicting Test Set
y_pred = voting_classifier.predict(X_test)
probability = estimator.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='micro')
rec = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='macro')
roc = roc_auc_score(y_test, probability, multi_class='ovr')
model_results = pd.DataFrame([['Ensemble Voting', acc, prec, rec, f1, roc]],
                             columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])

results = results.append(model_results, ignore_index = True)
proba.append(('Ensemble Voting',probability))
y_predict.append(('Ensemble Voting',list(y_pred)))

#The Best Classifier
the_best_estimator = results.sort_values(by='Accuracy',ascending=False).head(5)
print('The best classifier was:')
print('{}'.format(the_best_estimator))

y_pred = pd.DataFrame(y_predict, columns=['Model', 'y_pred'])
y_pred[y_pred['Model']==the_best_estimator.iloc[0,0]]['y_pred']
y_pred_final = y_pred[y_pred['Model']==the_best_estimator.iloc[0,0]]['y_pred']
y_pred_final = list(y_pred_final)
y_pred_final[0]

proba = pd.DataFrame(proba, columns=['Model', 'Proba'])
proba[proba['Model']==the_best_estimator.iloc[0,0]]['Proba']
proba_final = proba[proba['Model']==the_best_estimator.iloc[0,0]]['Proba']
proba_final = list(proba_final)
proba_final[0]


model_df = pd.DataFrame(model, columns=['Model', 'Estimator'])
model_df[model_df['Model']==the_best_estimator.iloc[0,0]]['Estimator']
model_df_final = model_df[model_df['Model']==the_best_estimator.iloc[0,0]]['Estimator']
model_df_final = list(model_df_final)
model_df_final[0]

training.ConfusionMatrixTito(y_test,y_pred_final[0])
training.ROCTito(model_df_final[0],X_train,X_test,y_train,y_test)
training.PrecisionRecallTito(model_df_final[0],X_train,X_test,y_train,y_test)
training.ClassPredictionErrorTito(model_df_final[0],X_train,X_test,y_train,y_test)
training.DiscriminationThresholdrTito(model_df_final[0],X_train,X_test,y_train,y_test)
training.LearningCurveTito(model_df_final[0], X_train,y_train, 'accuracy')
training.ValidationCurveTito(model_df_final[0], X_train,y_train, 'C', 'accuracy')

