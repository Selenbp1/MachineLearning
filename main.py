# 이름으로 성별 감지
import uvicorn
from fastapi import FastAPI
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
async def predict(name):
    vectorizer_name = gender_cv.transform([name]).toarray()
    prediction = gender_clf.predict(vectorizer_name)

    if prediction[0] == 0:
        result = "여성"
    else:
        result = "남성"

    return {'origin name' : name, "예측" : result}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
