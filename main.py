import dill
import zipfile

import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
try:
    with open('models/target_action_model.pkl', 'rb') as file:
        model = dill.load(file)
except FileNotFoundError:
    with zipfile.ZipFile('models/target_action_model.zip', 'r') as zip_ref:
        zip_ref.extractall(path='models/')
    with open('models/target_action_model.pkl', 'rb') as file:
        model = dill.load(file)


class Form(BaseModel):
    session_id: str
    client_id: float
    visit_date: str
    visit_time: str
    visit_number: int
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    utm_keyword: str
    device_category: str
    device_os: str
    device_brand: str
    device_model: str
    device_screen_resolution: str
    device_browser: str
    geo_country: str
    geo_city: str


class Prediction(BaseModel):
    session_id: str
    client_id: float
    is_target: int


@app.get('/status')
def status():
    return "I'm alive!"


@app.get('/version')
def version():
    return model['metadata']


@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    df = pd.DataFrame.from_dict([form.dict()])
    y = model['model'].predict(df)

    return {
        'session_id': form.session_id,
        'client_id': form.client_id,
        'Result': y[0]
    }
