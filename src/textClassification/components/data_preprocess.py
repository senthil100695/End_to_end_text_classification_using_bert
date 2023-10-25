import os
from textClassification.logging import logging
import pandas as pd
import nltk
import re
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import json
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from textClassification.entity import DataPreprocessConfig
from textClassification.utils.common import save_json

le = LabelEncoder()
stemmer = PorterStemmer()
lemat = WordNetLemmatizer()

class DataPreprocessor:
    def __init__(self, config: DataPreprocessConfig):
        self.config = config


    
    def feature_text_process(self,data_frame,feature):
        #remove stopwords and apply lemmatization for all data
        corpus =[]
        tags_re = re.compile('<.*?>')
        for i in range(len(data_frame[feature])):
            #text = BeautifulSoup(df['Full_Article'][i],'lxml').text
            text = re.sub(tags_re,'',data_frame[feature][i])
            text = re.sub('[^a-zA-Z]',' ',data_frame[feature][i])
            text = text.lower()
            text = text.split()
            review = [lemat.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
            review = ' '.join(review)
            review = review[1:len(review)-1]
            corpus.append(review)
        return corpus
        
    

    def convert(self):
        data_path = self.config.data_path
        
        #=================================
        df1 = pd.read_csv(data_path,encoding='cp1252')
        df1['Article_Type_encode'] = le.fit_transform(df1['Article_Type'].values)
        df1.drop(['Article.Banner.Image','Heading','Full_Article','Article.Description','Outlets','Id','Tonality'],axis=1,inplace=True)
        d1 = df1.drop_duplicates('Article_Type').set_index('Article_Type')
        d1 = d1.to_json()
        data_path_json = os.path.join(self.config.preprocess_data_path,'article_type.json')
        save_json(d1,data_path_json)
        #==================================

        #drop feature#
        df = pd.read_csv(data_path,encoding='cp1252')
        drop_feature = self.config.drop_feature
        df.drop(drop_feature,axis=1,inplace=True)
       
        
        #data Preprocssing and store preprocessed data into new column

        indepent_feature = self.config.indepent_feature
        preprocess_column = ['Heading_preprocess','Description_preprocess','Article_preprocess']
        for i in range(len(indepent_feature)):
            preprocess_feature= self.feature_text_process(df,indepent_feature[i])
            df[preprocess_column[i]] = preprocess_feature
        
        #convert cat_feature in numeric representation using labelencoder
        target_feature = self.config.target_feature
        df[target_feature]= le.fit_transform(df['Article_Type'])

        #train test split 

        X = df[preprocess_column]
        y = df['Article_Type']

        X_train, X_test , y_train, y_test = train_test_split(X,y, test_size = 0.20, random_state = 42)

        train = pd.concat([X_train,y_train],axis =1)
        test = pd.concat([X_test,y_test],axis =1)

        preprocess_path = os.path.join(self.config.preprocess_data_path)
        

        train.to_csv(preprocess_path+'train.csv')
        test.to_csv(preprocess_path+'test.csv')