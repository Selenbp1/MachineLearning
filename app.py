# python 가상환경 설정에 대한 문제 때문에 실행 오류가 발생하였음
# mac의 경우, 1) python3 -m venv env 가상환경 설치, 2) cd env 디렉토리 이동, 3) source bin/activate 실행하여 가상환경을 실행시켜줌
# Streamlit 시스템을 실행 시키기 위해서는 터미널에 streamlit run app.py 를 실행하면 됨

import streamlit as st
import joblib 
import time
from PIL import Image

gender_vectorizer = open("../../MachineLearning/model/gender_vectorizer.pkl", "rb")
gender_cv = joblib.load(gender_vectorizer)

gender_nv_model = open("../../MachineLearning/model/gender_nv_model.pkl", "rb")
gender_clf = joblib.load(gender_nv_model)

#Prediction
def predict_gender(data):
    vect = gender_cv.transform(data).toarray()
    result = gender_clf.predict(vect)
    return result

#Load CSS file
def load_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def load_icon(icon_name):
    st.markdown('<i class="material-icons">{}</i>'.format(icon_name), unsafe_allow_html=True)

#Images
def load_images(file_name):
    img = Image.open(file_name)
    return st.image(img, width=300)

def main():
    """Gender Classfier App
    Using Machine Learning and Streamlit
    
    """

    st.title("Gender Classfier")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
	<h2 style="color:white;text-align:center;">Streamlit ML App </h2>
	</div>

	"""
    st.markdown(html_temp, unsafe_allow_html=True)
    load_css('../../MachineLearning/icon.css')
    load_icon('people')

    name = st.text_input("Enter Name", "Type Here")
    if st.button("Predict"):
        result = predict_gender([name])
        if result[0] == 0:
            prediction = 'Female'
            c_img = '../../MachineLearning/female.png'
        else:
            result[0] == 1
            prediction = 'Male'
            c_img = '../../MachineLearning/male.png'

        st.success('Name: {} was classified as {}'.format(name.title(), prediction))
        load_images(c_img)
    
    if st.button("About"):
        st.text("SELENBP1")
        st.text("by BMK")


if __name__ == '__main__':
    main()