from textClassification.config.configuration import ConfigurationManager
import pickle
import re
import nltk
import re
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder


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
        review = ' '.join(review)
        review = review[1:len(review)-1]
        return review

    
    def predict(self,heading,article):
        heading_model_path = self.config.model_path+'heading_model.pkl'
        article_model_path = self.config.model_path+'article_model.pkl'

        heading_model = pickle.load(open(heading_model_path,"rb"))
        article_model = pickle.load(open(article_model_path,"rb"))

        heading_data = self.preprocess(heading)
        article_data = self.preprocess(article)

        sentence_model = SentenceTransformer('artifacts/base_model')
        head_emb = sentence_model.encode(heading_data)
        article_emb = sentence_model.encode(article_data)

        head_output = heading_model.predict(head_emb)
        article_output = article_model.predict(article_emb)

        return [head_output,article_output]