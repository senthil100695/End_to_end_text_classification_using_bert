import os
from textClassification.logging import logging
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from textClassification.exception import AppException
import sys
import numpy as np
from sklearn.metrics import accuracy_score
from textClassification.entity import (ModeltrainerConfig)
from textClassification.utils.common import save_json

class ModelTrainer:
    def __init__(self, config: ModeltrainerConfig):
        self.config = config
    
    def evaluate_models(self,X_train, y_train,X_test,y_test,models,param):

        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!! model name',i)
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            print('accuracy_score: ',accuracy_score(y_test,y_test_pred))
            #print('Precision: ',precision_score(y_test,y_test_pred))
            #print('Recall: ',recall_score(y_test,y_test_pred))
            #print('f1_score: ',f1_score(y_test,y_test_pred))


            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)
            test_accuracy_score = accuracy_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_accuracy_score

        return report


    def convert_text_vector(self):
        train_data = os.path.join(self.config.preprocess_data_path,'train.csv')
        test_data = os.path.join(self.config.preprocess_data_path,'test.csv')
        base_model_path = self.config.base_model_path
        sentence_model = SentenceTransformer(base_model_path)

        #for vectorization using sentence bert

        df_train = pd.read_csv(train_data)
        df_test = pd.read_csv(test_data)
        #target feature
        target_column = self.config.targert_feature
        y_train = df_train[target_column].to_list()
        y_test = df_test[target_column].to_list()

        #drop target feature
        df_train.drop([target_column],axis=1,inplace=True)
        df_test.drop([target_column],axis=1,inplace=True)

        input_feature = self.config.input_feature
        # embedding for train data
        heading_embedding = input_feature[0]
        description_embedding = input_feature[1]
        article_embedding = input_feature[2]

        heading_embedding = sentence_model.encode(df_train[heading_embedding])
        description_embedding = sentence_model.encode(df_train[description_embedding])
        article_embedding = sentence_model.encode(df_train[article_embedding])

        #test embedding 
        heading_emb = input_feature[0]
        description_emb = input_feature[1]
        article_emb = input_feature[2]

        head_test_emb = sentence_model.encode(df_test[heading_emb])
        desc_test_emb = sentence_model.encode(df_test[description_emb])
        art_test_emb = sentence_model.encode(df_test[article_emb]) 

        #model init
        print('@@@@@@@@@@@@@@ : model init')
        models = {
                'LogisticRegression':LogisticRegression(),
                 'SVM':SVC(),
                 'GaussianNB':GaussianNB(),
                 'SGDClassifier':SGDClassifier(),
                 'KNeighborsClassifier':KNeighborsClassifier(),
                 'DecisionTreeClassifier':DecisionTreeClassifier(),
                 'RandomForestClassifier':RandomForestClassifier(),
                 #'GradientBoostingClassifier':GradientBoostingClassifier(),
            }
        print('@@@@@@@@@@@@@@ : params init')
        params={
                "DecisionTreeClassifier": {
                    'max_depth':[3,5,7,10,15],
                    #'min_samples_leaf':[3,5,10,15,20],
                    #'min_samples_split':[8,10,12,18,20,16],
                    #'criterion':['gini','entropy'],
                },
                "RandomForestClassifier": { 
                    'n_estimators': [25, 50, 100, 150], 
                    #'max_features': ['sqrt', 'log2', None], 
                    #'max_depth': [3, 6, 9], 
                    #'max_leaf_nodes': [3, 6, 9] ,
                },
                #"GradientBoostingClassifier":{
                   # "n_estimators":[5,50,250,500],
                    #"max_depth":[1,3,5,7,9],
                    #"learning_rate":[0.01,0.1,1,10,100]
                #},
                "LogisticRegression":{
                    'penalty':['l2',None, 'elasticnet']},
                'GaussianNB':{
                    'var_smoothing': np.logspace(0,-9, num=100)},
                "KNeighborsClassifier":{ 
                    'n_neighbors' : [5,7,9,11,13,15],
                    #'weights' : ['uniform','distance'],
                    #'metric' : ['minkowski','euclidean','manhattan'] 
                     },
                "SGDClassifier":{
                    'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], # learning rate
                    #'penalty': ['l2'],
                    # 'n_jobs': [-1]
                    },
                'SVM' : {},

                }
                
 
        print('@@@@@@@@@@@@@@ : model_evaluation')
        model_report:dict=self.evaluate_models(X_train=heading_embedding,y_train=y_train,X_test=head_test_emb,y_test=y_test,
                                             models=models,param=params)
        
        print('@@@@@@@@@@@@@@ : model_evaluation complete')
        ## To get best model score from dict
        ## To get best model score from dict


        best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

        best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
        best_model = models[best_model_name]

        print('best_model_name: ',best_model_name,' best_model_score :' ,best_model_score)

        if best_model_score<0.6:
                print('No best model found')
        
        save_model_path = self.config.model_path+'heading_model.pkl'
        self.save_object(
                file_path=save_model_path,
                obj=best_model
            )

        predicted=best_model.predict(head_test_emb)

        accuracy = accuracy_score(y_test, predicted)

        print('@@@@@@@@@@@@@@ accuracy_score:', accuracy)

        model_metrics = {'Best Model Name' :best_model_name,
                         'Accuracy score': accuracy }
        
        head_model_metrics_path = os.path.join(self.config.model_path,'heading_metrics.json') 
        save_json(model_metrics,head_model_metrics_path)



        #article model########################################## 
        model_report:dict=self.evaluate_models(X_train=article_embedding,y_train=y_train,X_test=art_test_emb,y_test=y_test,
                                             models=models,param=params)

        best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

        best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
        best_model = models[best_model_name]

        print('best_model_name: ',best_model_name,' best_model_score :' ,best_model_score)

        if best_model_score<0.6:
                print('No best model found')
        
        save_model_path = self.config.model_path+'article_model.pkl'
        self.save_object(
                file_path=save_model_path,
                obj=best_model
            )

        predicted=best_model.predict(art_test_emb)

        accuracy = accuracy_score(y_test, predicted)
        print('@@@@@@@@@@@@@@ accuracy_score:', accuracy)

        model_metrics = {'Best Model Name' :best_model_name,
                         'Accuracy score': accuracy }
        
        article_model_metrics_path = os.path.join(self.config.model_path,'article_metrics.json') 
        save_json(model_metrics,article_model_metrics_path)






        #description model########################################## 
        model_report:dict=self.evaluate_models(X_train=description_embedding,y_train=y_train,X_test=desc_test_emb,y_test=y_test,
                                             models=models,param=params)

        best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

        best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
        best_model = models[best_model_name]

        print('best_model_name: ',best_model_name,' best_model_score :' ,best_model_score)

        if best_model_score<0.6:
                print('No best model found')
        
        save_model_path = self.config.model_path+'desc_model.pkl'
        self.save_object(
                file_path=save_model_path,
                obj=best_model
            )

        predicted=best_model.predict(desc_test_emb)

        accuracy = accuracy_score(y_test, predicted)
        print('@@@@@@@@@@@@@@ accuracy_score:', accuracy)

        model_metrics = {'Best Model Name' :best_model_name,
                         'Accuracy score': accuracy }
        
        description_model_metrics_path = os.path.join(self.config.model_path,'description_metrics.json') 
        save_json(model_metrics,description_model_metrics_path)

    
    def save_object(self,file_path, obj):
        try:
            dir_path = os.path.dirname(file_path)
            print('@@@@@@ model_path :',file_path)

            os.makedirs(dir_path, exist_ok=True)

            with open(file_path, "wb") as file_obj:
                pickle.dump(obj, file_obj)

        except Exception as e:
            raise AppException(e, sys)