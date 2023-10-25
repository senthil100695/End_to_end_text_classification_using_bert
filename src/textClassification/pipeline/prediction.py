from textClassification.config.configuration import ConfigurationManager
import pickle
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import json

le = LabelEncoder()
stemmer = PorterStemmer()
lemat = WordNetLemmatizer()
from sentence_transformers import SentenceTransformer


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_trainer_config()

    def preprocess(self,data):
        tags_re = re.compile('<.*?>')
        text = re.sub(tags_re,'',data)
        text = re.sub('[^a-zA-Z]',' ',data)
        text = text.lower()
        text = text.split()
        
        review = [lemat.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
        data_ = ' '.join(review)
        return data_

    
    def predict(self,heading,description,article):
        heading_model_path = self.config.model_path+'heading_model.pkl'
        article_model_path = self.config.model_path+'article_model.pkl'
        description_model_path = self.config.model_path+'desc_model.pkl'

        heading_model = pickle.load(open(heading_model_path,"rb"))
        description_model = pickle.load(open(description_model_path,"rb"))
        article_model = pickle.load(open(article_model_path,"rb"))

        heading_data = self.preprocess(heading)
        desc_data = self.preprocess(description)
        article_data = self.preprocess(article)


        if article_data[0]=='p' and article_data[-1]=='p':
            article_data = article_data[1:len(article_data)-1]

        if desc_data[0]=='p' and desc_data[-1]=='p':
            desc_data = desc_data[1:len(desc_data)-1]
        

        sentence_model = SentenceTransformer('artifacts/base_model')
        head_emb = sentence_model.encode([heading_data])
        des_emb = sentence_model.encode([desc_data])
        article_emb = sentence_model.encode([article_data])


        head_output = heading_model.predict(head_emb)
        desc_output = description_model.predict(des_emb)
        article_output = article_model.predict(article_emb)

        #check the class type
        article_cls = open('F:\\senthil\\project\\End_to_end_text_classification_using_bert\\research\\article_type.json')
        data = json.load(article_cls)

        key_list = list(data['Article_Type_encode'].keys())
        val_list = list(data['Article_Type_encode'].values())
        
        hed_pos = val_list.index(head_output[0])
        if hed_pos == val_list.index(article_output[0]) or hed_pos == val_list.index(desc_output[0]) :
            return key_list[hed_pos]

    
if __name__=="__main__":
    heading:str = "NTSB Report Provides New Details in Southeast Alaska Helicopter Crash That Killed 3"
    article:str ='<p>The helicopter that crashed in Southeast Alaska in late September, killing three people, entered a 500-foot freefall before dropping to a Glacier Bay National Park beach, according to by the National Transportation Safety Board.&nbsp;The preliminary NTSB report released Friday offers no official probable cause. That determination won&lsquo;t be made until next year at the earliest.</p>'
    obj = PredictionPipeline()
    score = obj.predict(heading,article)
    print(score)