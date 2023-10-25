
from flask import Flask, render_template, request, url_for, redirect,Response
from textClassification.pipeline.prediction import PredictionPipeline
import os

 

app = Flask(__name__, template_folder='Flask_templates\\templates', static_folder='Flask_templates\\static')
 

 
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        heading = request.form["Article Heading"]
        description = request.form["Article Description"]
        article = request.form["Full Article"]


        return redirect(url_for('result', name=result))
    return render_template('home.html')

@app.route("/result", methods=['GET', 'POST'])
def result():
    heading = request.form.get('heading')
    description = request.form.get('description')
    article = request.form.get('article')

    
    obj = PredictionPipeline()
    result = obj.predict(heading,description,article)
    name = result
    print('@@@@@@@@@@@@@@@@@',result)
    #return render_template('index.html',result=result)

    return render_template("result.html", name=name)

@app.get("/train")
def training():
        os.system("python main.py")
        return Response("Training successful !!")

if __name__=='__main__':
    app.run(debug = True)