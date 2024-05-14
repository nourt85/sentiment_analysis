import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pipeline

app = Flask(__name__)

pkl_lr_filename = 'BOW_lr_model.pkl'
pkl_BOW_filename = 'BOW.pkl'
with open(pkl_lr_filename, 'rb') as file:
    lr = pickle.load(file)
with open(pkl_BOW_filename, 'rb') as file:
    bow = pickle.load(file)
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    #int_features = [float(x) for x in request.form.values()]
    text = request.form['review_text']
    norm_text = pipeline.normalize_corpus(np.array([text]), html_stripping=True, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True,
                     text_lemmatization=True, special_char_removal=True,
                     stopword_removal=True)
    feat=bow.transform(norm_text)
    sentiment = lr.predict(feat)[0]
    confidence = lr.predict_proba(feat)[0][sentiment]
    #sentiment=0
    return render_template('index.html',
                           prediction_text=("Positive!" if (sentiment == 1) else "Negative!"),
                           confidence_score="confidence: {0:.0%}".format(confidence),
                           review_text=text,
                           )


if __name__ == "__main__":
    app.run(debug=True)
