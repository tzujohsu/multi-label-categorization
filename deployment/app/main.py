# fastapi dev main.py
from fastapi import FastAPI
from app.functions import returnClassifyResults
import joblib


# define model info
model_name = 'trained_model.joblib'
model_path = "app/data/" + model_name

# load model
model = joblib.load(model_path)

# label name
label_names = ['crime, law and justice', 'arts, culture, entertainment and media',
       'economy, business and finance',
       'disaster, accident and emergency incident', 'environment',
       'education', 'health', 'human interest', 'lifestyle and leisure',
       'politics', 'labour', 'religion and belief',
       'science and technology', 'society', 'sport',
       'conflict, war and peace', 'weather']

# create FastAPI object
app = FastAPI()

# API operations
@app.get("/")
def health_check():
    return {'health_check': 'OK'}

@app.get("/info")
def info():
    return {'name': 'multi-label-content-cateorization', 
            'description': "Categorization API for Tzu-Jo Hsu's NLP & ML project demo."}

@app.get("/categorize")
def categorize(txt: str):
    topk_indices, topk_probabilities = returnClassifyResults(txt, model)
    topk_categories = [label_names[i] for i in topk_indices]
    response = {category: probability for category, probability in zip(topk_categories, topk_probabilities)}
    return response