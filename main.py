# 이름으로 성별 감지
from unittest import result
import uvicorn
from fastapi import FastAPI, Query
import joblib

gender_vectorizer = open("model/gender_vectorizer.pkl", "rb")
gender_cv = joblib.load(gender_vectorizer)

gender_nv_model = open("model/gender_nv_model.pkl", "rb")
gender_clf = joblib.load(gender_nv_model)

app = FastAPI()

@app.get('/')
async def index():
    return {'message': 'Hi'}

@app.get('/items/{name}')
async def get_items(name):
    return {'name':name}

@app.get('/predict')
async def predict(name:str = Query(None, min_length=2, max_length=12)):
    vectorizer_name = gender_cv.transform([name]).toarray()
    prediction = gender_clf.predict(vectorizer_name)

    if prediction[0] == 0:
        result = "female"
    else:
        result = "male"

    return {'origin name' : name, "predict" : result}

@app.post('/predict/{name}')
async def predict(name):
    vectorized_name = gender_cv.transform([name]).toarray()
    prediction = gender_clf.predict(vectorized_name)
    if prediction[0] == 0:
        result = "female"
    else:
        result = "male"
    
    return {"origin name" : name, "predict":result}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
