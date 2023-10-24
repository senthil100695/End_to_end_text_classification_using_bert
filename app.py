from fastapi import FastAPI
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from textClassification.pipeline.prediction import PredictionPipeline


heading:str = "NTSB Report Provides New Details in Southeast Alaska Helicopter Crash That Killed 3"
article:str ='<p>The helicopter that crashed in Southeast Alaska in late September, killing three people, entered a 500-foot freefall before dropping to a Glacier Bay National Park beach, according to by the National Transportation Safety Board.&nbsp;The preliminary NTSB report released Friday offers no official probable cause. That determination won&lsquo;t be made until next year at the earliest.</p>'

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")



@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    



@app.post("/predict")
async def predict_route(heading,article):
    try:
        obj = PredictionPipeline()
        text = obj.predict(heading,article)
        return text
    except Exception as e:
        raise e
    

if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
