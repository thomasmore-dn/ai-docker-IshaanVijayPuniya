from flask import Flask, request,render_template, json
import joblib
import pandas as pd
from joblib import load
import pandas as pd
import spacy
import re
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

@app.route('/',methods=["Get","POST"])
def home():
    return render_template("index.html")

@app.route('/predict',methods=["Get","POST"])
def predict():
    plot = request.form["plot"]
    cleanplot = data_validation(plot)
    #with open("model1.pkl", 'rb') as file:
    #        classifier = joblib.load(file)
    classifier = load('/app/Model-File/model.joblib')        
    #xgb_clf = joblib.load('xgb_clf.pkl')

    predictions = classifier.predict_proba(cleanplot)
    columns = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
               'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

    ranked_predictions = (-predictions).argsort(axis=1)[:, :2]  # getting the indexes of the predicitons.
    ranked_genre_predictions = [[columns[index] for index in row] for row in ranked_predictions]
    # Convert NumPy array to Python list
   

    return json.dumps(ranked_genre_predictions)

def data_validation(plot_text: str):
    clean = []
    dialog = re.sub(pattern='[^a-zA-Z\s]', repl='', string=plot_text)
    dialog = dialog.lower()
    words = nlp(dialog)
    dialog_words = [token.lemma_ for token in words if not token.is_stop]
    dialog = ' '.join(dialog_words)
    clean.append(dialog)
    return clean
    

if __name__ == '__main__':
    app.run(debug=True, port=5002)
