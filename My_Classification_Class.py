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
from yellowbrick.classifier import ROCAUC, PrecisionRecallCurve, ClassPredictionError, DiscriminationThreshold
from yellowbrick.model_selection import ValidationCurve, LearningCurve
import time
from sklearn.model_selection import StratifiedKFold

class MyClassificationClass:
    '''
    Class created by Tito with propose to train a classification machine learning process.
    '''
    def __init__(self,stratify,test_size=0.2, random_state=0, binary=True):
        '''
        Parameters Initialization
        Parameters
    ----------
    stratify: array-like, default= y
    If not None, data is split in a stratified fashion, using this as the class labels. Read more in the User Guide.
    
    random_state: int, RandomState instance or None, default=0
    Controls the shuffling applied to the data before applying the split. 
    Pass an int for reproducible output across multiple function calls.
    
    binary: True for binary classfication or False for multiclassificatio, default=True.
        '''
        self.test_size = test_size
        self.stratify = stratify
        self.random_state=random_state
        self.binary = binary
        
    def TrainTestSet(self,X,y):
        X_train,X_test,y_train,y_test = train_test_split(X, y , test_size=self.test_size,
                                                                             stratify=self.stratify, random_state=0)
        return X_train,X_test,y_train,y_test
        
    def LogisticRegressionTito(self,X_train,X_test,y_train,y_test):
        ''' Logistic Regression'''
        if self.binary == True:
            lr_classifier = LogisticRegression(random_state = 0, penalty = 'l2')
            lr_classifier.fit(X_train, y_train)    
            y_pred = lr_classifier.predict(X_test)
            proba = lr_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_pred)
            results = pd.DataFrame([['Logistic Regression (Lasso)', acc, prec, rec, f1,roc]],
                                     columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            return results, y_pred , proba, lr_classifier
        
        if self.binary == False:
            lr_classifier = LogisticRegression(random_state = 0, penalty = 'l2', solver="liblinear")
            lr_classifier.fit(X_train, y_train)    
            y_pred = lr_classifier.predict(X_test)
            proba = lr_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='micro')
            rec = recall_score(y_test, y_pred, average='micro')
            f1 = f1_score(y_test, y_pred, average='macro')
            roc = roc_auc_score(y_test, proba, multi_class='ovr')
            results = pd.DataFrame([['Logistic Regression (Lasso)', acc, prec, rec, f1,roc]],
                                     columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            return results, y_pred, proba , lr_classifier
        
        if self.binary != False or self.binary != True:
            raise ValueError("The 'binary' value on MyClassificationClass Must be True or False")
            
    def KnnClassifierTito(self,X_train,X_test,y_train,y_test):
        ## K-Nearest Neighbors (K-NN)
        #Choosing the K value
        error_rate= []
        for i in np.arange(1,40):
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            pred_i = knn.predict(X_test)
            error_rate.append(np.mean(pred_i != y_test))
        plt.figure(figsize=(10,6))
        plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
                 markerfacecolor='red', markersize=10)
        plt.title('Error Rate vs. K Value')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        n_neighbors = pd.DataFrame({'K':np.arange(1,40),
                                    'Error_Rate':error_rate})
        n_neighbors = n_neighbors['Error_Rate'].argmin()
        print("The smallest error was with 'K' = {}".format(n_neighbors))
        
        if self.binary == True:
            kn_classifier = KNeighborsClassifier(n_neighbors= n_neighbors, metric='minkowski', p= 2)
            kn_classifier.fit(X_train, y_train)    
            y_pred = kn_classifier.predict(X_test)
            proba = kn_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_pred)
            results = pd.DataFrame([['K-Nearest Neighbors (minkowski)', acc, prec, rec, f1, roc]],
                                     columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            return results, y_pred , proba, kn_classifier
        
        if self.binary == False:
            kn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric='minkowski', p= 2)
            kn_classifier.fit(X_train, y_train)    
            y_pred = kn_classifier.predict(X_test)
            proba = kn_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='micro')
            rec = recall_score(y_test, y_pred, average='micro')
            f1 = f1_score(y_test, y_pred, average='micro')
            roc = roc_auc_score(y_test, proba, multi_class='ovr')
            results = pd.DataFrame([['K-Nearest Neighbors (minkowski)', acc, prec, rec, f1, roc]],
                                     columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            return results, y_pred, proba , kn_classifier
        
        if self.binary != False or self.binary != True:
            raise ValueError("The 'binary' value on MyClassificationClass/n"
                             "Must be True or False")        
            
    def SuportVectorLinearTito(self,X_train,X_test,y_train,y_test):
        '''SVM (Linear)'''
        if self.binary == True:
            svm_linear_classifier = SVC(random_state = 0, kernel = 'linear', probability= True)
            svm_linear_classifier.fit(X_train, y_train)    
            y_pred = svm_linear_classifier.predict(X_test)
            proba = svm_linear_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_pred)
            results = pd.DataFrame([['SVM (Linear)', acc, prec, rec, f1, roc]],
                                     columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            return results, y_pred , proba, svm_linear_classifier
        
        if self.binary == False:
            svm_linear_classifier = SVC(random_state = 0, kernel = 'linear', probability= True)
            svm_linear_classifier.fit(X_train, y_train)    
            y_pred = svm_linear_classifier.predict(X_test)
            proba = svm_linear_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='micro')
            rec = recall_score(y_test, y_pred, average='micro')
            f1 = f1_score(y_test, y_pred, average='macro')
            roc = roc_auc_score(y_test, proba, multi_class='ovr')
            results = pd.DataFrame([['SVM (Linear)', acc, prec, rec, f1, roc]],
                                     columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            return results, y_pred, proba , svm_linear_classifier
        
        if self.binary != False or self.binary != True:
            raise ValueError("The 'binary' value on MyClassificationClass Must be True or False")
            
    def SuportVectorRBFTito(self,X_train,X_test,y_train,y_test):
        '''SVM (rbf)'''
        if self.binary == True:
            svm_rbf_classifier = SVC(random_state = 0, kernel = 'rbf', probability= True)
            svm_rbf_classifier.fit(X_train, y_train)    
            y_pred = svm_rbf_classifier.predict(X_test)
            proba = svm_rbf_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_pred)
            results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1, roc]],
                                     columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            return results, y_pred , proba, svm_rbf_classifier
        
        if self.binary == False:
            svm_rbf_classifier = SVC(random_state = 0, kernel = 'rbf', probability= True)
            svm_rbf_classifier.fit(X_train, y_train)    
            y_pred = svm_rbf_classifier.predict(X_test)
            proba = svm_rbf_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='micro')
            rec = recall_score(y_test, y_pred, average='micro')
            f1 = f1_score(y_test, y_pred, average='macro')
            roc = roc_auc_score(y_test, proba, multi_class='ovr')
            results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1, roc]],
                                     columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            return results, y_pred, proba , svm_rbf_classifier
        
        if self.binary != False or self.binary != True:
            raise ValueError("The 'binary' value on MyClassificationClass Must be True or False") 

    def NaiveBayesTito(self,X_train,X_test,y_train,y_test):
        '''Naive Bayes'''
        if self.binary == True:
            gb_classifier = GaussianNB()
            gb_classifier.fit(X_train, y_train)    
            y_pred = gb_classifier.predict(X_test)
            proba = gb_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_pred)
            results = pd.DataFrame([['Naive Bayes (Gaussian)', acc, prec, rec, f1, roc]],
                                     columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            return results, y_pred , proba, gb_classifier
        
        if self.binary == False:
            gb_classifier = GaussianNB()
            gb_classifier.fit(X_train, y_train)    
            y_pred = gb_classifier.predict(X_test)
            proba = gb_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='micro')
            rec = recall_score(y_test, y_pred, average='micro')
            f1 = f1_score(y_test, y_pred, average='macro')
            roc = roc_auc_score(y_test, proba, multi_class='ovr')
            results = pd.DataFrame([['Naive Bayes (Gaussian)', acc, prec, rec, f1, roc]],
                                     columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            return results, y_pred, proba , gb_classifier
        
        if self.binary != False or self.binary != True:
            raise ValueError("The 'binary' value on MyClassificationClass Must be True or False")             

    def DecisionTreeTito(self,X_train,X_test,y_train,y_test):
        '''Decision Tree'''
        if self.binary == True:
            dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
            dt_classifier.fit(X_train, y_train)    
            y_pred = dt_classifier.predict(X_test)
            proba = dt_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_pred)
            results = pd.DataFrame([['Decision Tree', acc, prec, rec, f1, roc]],
                                   columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            plt.figure(figsize=(25,20))
            tree.plot_tree(dt_classifier, 
                   feature_names=X_train.columns,  
                   class_names=[str(np.unique(y_train.values)[i]) for i in np.arange(len(np.unique(y_train.values)))],
                   filled=True)
            print('The model was trained.')
            return results, y_pred , proba, dt_classifier
        
        if self.binary == False:
            dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
            dt_classifier.fit(X_train, y_train)    
            y_pred = dt_classifier.predict(X_test)
            proba = dt_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='micro')
            rec = recall_score(y_test, y_pred, average='micro')
            f1 = f1_score(y_test, y_pred, average='macro')
            roc = roc_auc_score(y_test, proba, multi_class='ovr')
            results = pd.DataFrame([['Decision Tree', acc, prec, rec, f1, roc]],
                                   columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            plt.figure(figsize=(25,20))
            tree.plot_tree(dt_classifier, 
                   feature_names=X_train.columns,  
                   class_names=[str(np.unique(y_train.values)[i]) for i in np.arange(len(np.unique(y_train.values)))],
                   filled=True)
            print('The model was trained.')
            return results, y_pred, proba , dt_classifier
        
        if self.binary != False or self.binary != True:
            raise ValueError("The 'binary' value on MyClassificationClass Must be True or False")            
    
    def RandomForestTito(self,X_train,X_test,y_train,y_test):
        '''Random Forest'''
        if self.binary == True:
            rf_classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,
                                    criterion = 'gini')
            rf_classifier.fit(X_train, y_train)    
            y_pred = rf_classifier.predict(X_test)
            proba = rf_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_pred)
            results = pd.DataFrame([['Random Forest Gini (n=100)', acc, prec, rec, f1, roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            return results, y_pred , proba, rf_classifier
        
        if self.binary == False:
            rf_classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,
                                    criterion = 'gini')
            rf_classifier.fit(X_train, y_train)    
            y_pred = rf_classifier.predict(X_test)
            proba = rf_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='micro')
            rec = recall_score(y_test, y_pred, average='micro')
            f1 = f1_score(y_test, y_pred, average='macro')
            roc = roc_auc_score(y_test, proba, multi_class='ovr')
            results = pd.DataFrame([['Random Forest Gini (n=100)', acc, prec, rec, f1, roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            return results, y_pred, proba , rf_classifier
        
        if self.binary != False or self.binary != True:
            raise ValueError("The 'binary' value on MyClassificationClass Must be True or False")
            
    def AdaBoostTito(self,X_train,X_test,y_train,y_test):
        '''Ada Boosting'''
        if self.binary == True:
            ad_classifier = AdaBoostClassifier()
            ad_classifier.fit(X_train, y_train)    
            y_pred = ad_classifier.predict(X_test)
            proba = ad_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_pred)
            results = pd.DataFrame([['Ada Boosting', acc, prec, rec, f1, roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            return results, y_pred , proba, ad_classifier
        
        if self.binary == False:
            ad_classifier = AdaBoostClassifier()
            ad_classifier.fit(X_train, y_train)    
            y_pred = ad_classifier.predict(X_test)
            proba = ad_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='micro')
            rec = recall_score(y_test, y_pred, average='micro')
            f1 = f1_score(y_test, y_pred, average='macro')
            roc = roc_auc_score(y_test, proba, multi_class='ovr')
            results = pd.DataFrame([['Ada Boosting', acc, prec, rec, f1, roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            return results, y_pred, proba , ad_classifier
        
        if self.binary != False or self.binary != True:
            raise ValueError("The 'binary' value on MyClassificationClass Must be True or False")
    
    def GradientBoostTito(self,X_train,X_test,y_train,y_test):
        '''Gradient Boosting'''
        if self.binary == True:
            gr_classifier = GradientBoostingClassifier()
            gr_classifier.fit(X_train, y_train)    
            y_pred = gr_classifier.predict(X_test)
            proba = gr_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_pred)
            results = pd.DataFrame([['Gradient Boosting', acc, prec, rec, f1, roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            return results, y_pred , proba, gr_classifier
        
        if self.binary == False:
            gr_classifier = GradientBoostingClassifier()
            gr_classifier.fit(X_train, y_train)    
            y_pred = gr_classifier.predict(X_test)
            proba = gr_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='micro')
            rec = recall_score(y_test, y_pred, average='micro')
            f1 = f1_score(y_test, y_pred, average='macro')
            roc = roc_auc_score(y_test, proba, multi_class='ovr')
            results = pd.DataFrame([['Gradient Boosting', acc, prec, rec, f1, roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            return results, y_pred, proba , gr_classifier
        
        if self.binary != False or self.binary != True:
            raise ValueError("The 'binary' value on MyClassificationClass Must be True or False")

    def XgBoostTito(self,X_train,X_test,y_train,y_test):
        '''Xg Boosting'''
        if self.binary == True:
            xg_classifier = XGBClassifier()
            xg_classifier.fit(X_train, y_train)    
            y_pred = xg_classifier.predict(X_test)
            proba = xg_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_pred)
            results = pd.DataFrame([['Xg Boosting', acc, prec, rec, f1, roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            return results, y_pred , proba, xg_classifier
        
        if self.binary == False:
            xg_classifier = XGBClassifier()
            xg_classifier.fit(X_train, y_train)    
            y_pred = xg_classifier.predict(X_test)
            proba = xg_classifier.predict_proba(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='micro')
            rec = recall_score(y_test, y_pred, average='micro')
            f1 = f1_score(y_test, y_pred, average='macro')
            roc = roc_auc_score(y_test, proba, multi_class='ovr')
            results = pd.DataFrame([['Xg Boosting', acc, prec, rec, f1, roc]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC Score'])
            return results, y_pred, proba , xg_classifier
        
        if self.binary != False or self.binary != True:
            raise ValueError("The 'binary' value on MyClassificationClass Must be True or False")
    
    def ConfusionMatrixTito(self, y_test, y_pred):
        if self.binary == True:
            cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction
            df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
            plt.figure(figsize = (10,7))
            sns.set(font_scale=1.4)
            sns.heatmap(df_cm, annot=True, fmt='g')
            print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
            return df_cm
        
        if self.binary == False:
            cm = multilabel_confusion_matrix(y_test,y_pred, labels= y_test.unique())
            labels = ['Positive', 'Negative']
            ind = np.arange(len(labels))
            fig = plt.figure(figsize=(15,15))
            for i in np.arange(1,len(cm)+1):
                ax = fig.add_subplot(3,3,i)
                ax.set_title('Confusion Matrix for Class {}'.format(i))
                ax.imshow(cm[i-1], cmap='viridis')
                for x in ind:
                    for j in ind:
                        text = ax.text(j, x, cm[i-1][x, j],ha="center", va="center", color="r")
                        ax.set_xticks(ind)
                        ax.set_yticks(ind)
                        ax.set_xticklabels(labels)
                        ax.set_yticklabels(labels)
                        ax.xaxis.tick_top()
                        plt.tight_layout()  
            plt.show()
            return cm
            
        if self.binary != False or self.binary != True:
            raise ValueError("The 'binary' value on MyClassificationClass/n"
                             "Must be True or False")
            
    def ROCTito(self,model,X_train,X_test,y_train,y_test):
        visualizer = ROCAUC(model, classes=[str(np.unique(y_train.values)[i]) for i in np.arange(len(np.unique(y_train.values)))])
        visualizer.fit(X_train, y_train)        # Fit the training data to the visualizer
        visualizer.score(X_test, y_test)        # Evaluate the model on the test data
        visualizer.show()                       # Finalize and show the figure
            
    def PrecisionRecallTito(self,model,X_train,X_test,y_train,y_test):
        viz = PrecisionRecallCurve(
                model,
                classes=[str(np.unique(y_train.values)[i]) for i in np.arange(len(np.unique(y_train.values)))],
                iso_f1_curves=True,
                per_class=True,
                micro=False
                )
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.show()
       
    def ClassPredictionErrorTito(self,model,X_train,X_test,y_train,y_test):
        visualizer = ClassPredictionError(
                model,
                classes=[str(np.unique(y_train.values)[i]) for i in np.arange(len(np.unique(y_train.values)))]
                )
        visualizer.fit(X_train, y_train) # Fit the training data to the visualizer
        visualizer.score(X_test, y_test) # Evaluate the model on the test data
        visualizer.show() # Draw visualization  
        
    def DiscriminationThresholdrTito(self,model,X_train,X_test,y_train,y_test):
        if self.binary ==True:
            visualizer = DiscriminationThreshold(model, n_trials=10)
            visualizer.fit(X_train, y_train)        # Fit the data to the visualizer
            visualizer.show()           # Finalize and render the figure
            print('A visualization of precision, recall, f1 score, and queue rate with respect to the discrimination threshold of a binary classifier.')
            print('The discrimination threshold is the probability or score at which the positive class is chosen over the negative class.') 
            print('Generally, this is set to 50% but the threshold can be adjusted to increase or decrease the sensitivity to false positives or to other application factors.')
        else:
            print('This visualizer only works for binary classification.')
    
    def LearningCurveTito(self,model, X_train, y_train, scoring=str):
        '''
        For attribute scoring consult the page https://scikit-learn.org/stable/modules/model_evaluation.html
        '''
        cv = StratifiedKFold(n_splits=12)
        sizes = np.linspace(0.3, 1.0, 10)
        visualizer = LearningCurve(
                model, cv=cv, scoring=scoring, train_sizes=sizes, n_jobs=4
                )

        visualizer.fit(X_train, y_train)        # Fit the data to the visualizer
        visualizer.show()           # Finalize and render the figure
    
    def ValidationCurveTito(self,model, X_train, y_train, param_name=str, scoring=str):
        '''
        For attribute param_name consult the page https://www.scikit-yb.org/en/latest/api/model_selection/validation_curve.html
        For attribute scoring consult the page https://scikit-learn.org/stable/modules/model_evaluation.html
        '''
        start = time.time()
        viz = ValidationCurve(
                model, param_name=param_name,
                param_range=np.arange(1, 11), cv=10, scoring=scoring
                )
        viz.fit(X_train, y_train)
        viz.show()
        end = time.time()
        print("The creation of Validation Curve graph has taken {}".format(time.strftime("%H:%M:%S",time.gmtime(end-start))))
      
    def AllEstimatorsTito(self,X_train,X_test,y_train,y_test):
        '''
        'lr', lr_classifier,
        'kn', kn_classifier,
        'svc_linear', svm_linear_classifier,
        'svc_rbf', svm_rbf_classifier,
        'gb', gb_classifier,
        'dt', dt_classifier,
        'rf', rf_classifier,
        'ad', ad_classifier,
        'gr', gr_classifier,
        'xg', xg_classifier,
        '''
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
        
        if self.binary == True:
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
                                                  ('xg', xg_classifier),], voting='soft')
            
            for clf in (lr_classifier,kn_classifier,svm_linear_classifier,svm_rbf_classifier,
                        gb_classifier, dt_classifier,rf_classifier, ad_classifier, gr_classifier, xg_classifier,
                        voting_classifier):
                clf.fit(X_train,y_train)
                y_pred = clf.predict(X_test)
                print(clf.__class__.__name__, accuracy_score(y_test, y_pred))

            # Predicting Test Set
            y_pred = voting_classifier.predict(X_test)
            probability = voting_classifier.predict_proba(X_test)
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
            y_pred = y_pred[y_pred['Model']==the_best_estimator.iloc[0,0]]['y_pred']
            y_pred = list(y_pred)
            
            proba = pd.DataFrame(proba, columns=['Model', 'Proba'])
            proba = proba[proba['Model']==the_best_estimator.iloc[0,0]]['Proba']
            proba = list(proba)
        
            model_df = pd.DataFrame(model, columns=['Model', 'Estimator'])
            estimator = model_df[model_df['Model']==the_best_estimator.iloc[0,0]]['Estimator']
            estimator = list(estimator)
            
            return results, y_pred[0] , proba[0], estimator[0]
    
        if self.binary == False:
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
            probability = voting_classifier.predict_proba(X_test)
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
            y_pred = y_pred[y_pred['Model']==the_best_estimator.iloc[0,0]]['y_pred']
            y_pred = list(y_pred)
            
            proba = pd.DataFrame(proba, columns=['Model', 'Proba'])
            proba = proba[proba['Model']==the_best_estimator.iloc[0,0]]['Proba']
            proba = list(proba)
        
            model_df = pd.DataFrame(model, columns=['Model', 'Estimator'])
            estimator = model_df[model_df['Model']==the_best_estimator.iloc[0,0]]['Estimator']
            estimator = list(estimator)
            
            return results, y_pred[0] , proba[0], estimator[0]
        
        if self.binary != False or self.binary != True:
            raise ValueError("The 'binary' value on MyClassificationClass Must be True or False")
