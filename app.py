from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from textClassification.pipeline.prediction import PredictionPipeline
from pydantic import BaseModel

class Article(BaseModel):
    heading: str
    article_data: str



application = FastAPI()

#@app.get("/")
#async def index():
#    return RedirectResponse(url="/docs")
@application.get("/")
def index():
    return {'message': 'Welcome Home'}



@application.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    

@application.get('/welcome')
def get_name(name:str):
    return {"welcome to my world":f'{name}'}

@application.post("/predict")
def predict_route(data:Article):
    try:
        data = data.dict()
        print(data)
        heading = data['heading']
        article_data_ = data['article_data']
        obj = PredictionPipeline()
        text = obj.predict(heading,article_data_)
        return text
    except Exception as e:
        raise e
    

if __name__=="__main__":
    #uvicorn.run(app, host="0.0.0.0", port=8080)
    uvicorn.run(application, host="127.0.0.1", port=8000)
#uvicorn app:application --reload