import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# Set page configurations
st.set_page_config(page_title="Health Guard", layout="wide", initial_sidebar_state="expanded")

# Define CSS styling to make the UI more attractive
st.markdown("""
    <style>
    .title {
        color: #1F77B4;
        font-weight: bold;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #E9F7EF;
    }
    .stButton>button {
        color: white;
        background-color: #1F77B4;
        width: 100%;
        font-size: 16px;
    }
    .result-text {
        font-size: 24px;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Load models only once with Streamlit's caching
@st.cache_resource
def load_models():
    diabetes_model = pickle.load(open('Diabetes.pkl', 'rb'))
    heart_model = pickle.load(open('heart.pkl', 'rb'))
    parkinsons_model = pickle.load(open('parkinsons.pkl', 'rb'))
    return diabetes_model, heart_model, parkinsons_model

diabetes_model, heart_model, parkinsons_model = load_models()

# Sidebar for navigation with icons
with st.sidebar:
    st.image("https://static7.depositphotos.com/1007989/773/i/450/depositphotos_7735215-stock-illustration-health-is-wealth.jpg", width=120)  # Replace with an image URL or path
    st.markdown("<h2 style='text-align: center; color: #1F77B4;'>Health Guard</h2>", unsafe_allow_html=True)
    selected = option_menu(
        'Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', 'Parkinsons Prediction'],
        menu_icon='hospital-fill',
        icons=['activity', 'heart', 'person'],
        styles={
            "container": {"padding": "0!important"},
            "icon": {"color": "#1F77B4", "font-size": "25px"}, 
            "nav-link": {"font-size": "18px", "text-align": "center", "margin": "5px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#1F77B4"},
        }
    )

# Diabetes Prediction
if selected == 'Diabetes Prediction':
    st.markdown("<h2 class='title'>Diabetes Prediction Model</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    glucose = col1.slider('Glucose Level', 0, 500, 120, help="Normal range: 90-130 mg/dL")
    bp = col2.slider('Blood Pressure Level', 0, 200, 120, help="Normal range: 80-120 mmHg")
    skthic = col3.slider('Skin Thickness Value', 0, 100, 20)
    insulin = col1.slider('Insulin Level', 0, 900, 30, help="Normal range: 70-120 pmol/L")
    bmi = col2.slider('BMI Value', 0.0, 70.0, 25.0)
    dpf = col3.slider('Diabetes Pedigree Function Value', 0.0, 2.5, 0.5)
    age = col1.slider('Age of the Person', 0, 100, 25)

    if st.button('Get Diabetes Test Result'):
        user_input = [glucose, bp, skthic, insulin, bmi, dpf, age]
        pred = diabetes_model.predict([user_input])[0]
        diab_diagnosis = 'The person is Diabetic 🩸' if pred == 1 else 'The person is not Diabetic 👍'
        st.markdown(f"<p class='result-text'>{diab_diagnosis}</p>", unsafe_allow_html=True)

# Heart Disease Prediction
if selected == 'Heart Disease Prediction':
    st.markdown("<h2 class='title'>Heart Disease Prediction Using ML</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    age = col1.slider('Age', 0, 100, 50)
    gender = col1.radio('Gender', ['Male', 'Female'])
    cp = col3.selectbox('Chest Pain Types', ['Type1', 'Type2', 'Type3', 'Type4'])
    trestbps = col1.slider('Resting Blood Pressure', 0, 200, 120)
    chol = col2.slider('Serum Cholesterol in mg/dl', 50, 600, 200)
    fbs = col3.radio('Fasting Blood Sugar >120', ['Yes', 'No'])
    restecg = col1.radio('Resting Electrocardiograph Results', ['Normal', 'Abnormal'])
    mhra = col2.slider('Maximum Heart Rate Achieved', 50, 200, 80)
    ang = col3.radio('Exercise Induced Angina', ['Yes', 'No'])
    oldpeak = col1.slider('ST depression induced by exercise', 0.0, 10.0, 1.0)
    slope = col2.selectbox('Slope of the peak exercise ST segment', ['UPsloping', 'Flat', 'Downsloping'])
    cf = col3.slider('Major Vessels colored by fluoroscopy', 0, 4, 0)
    thal = col1.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

    cp_mapping = {'Type1': 0, 'Type2': 1, 'Type3': 2, 'Type4': 3}
    slope_mapping = {'UPsloping': 0, 'Flat': 1, 'Downsloping': 2}
    thal_mapping = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}

    if st.button('Get Heart Disease Test Result'):
        user_input = [
            age, 1 if gender == 'Male' else 0, cp_mapping[cp], trestbps, chol,
            1 if fbs == 'Yes' else 0, 1 if restecg == 'Normal' else 0, mhra,
            1 if ang == 'Yes' else 0, oldpeak, slope_mapping[slope], cf, thal_mapping[thal]
        ]
        pred = heart_model.predict([user_input])[0]
        heart_diagnosis = 'The person has Heart Disease 💔' if pred == 1 else 'The person does not have Heart Disease ❤️'
        st.markdown(f"<p class='result-text'>{heart_diagnosis}</p>", unsafe_allow_html=True)

# Parkinson's Disease Prediction
if selected == 'Parkinsons Prediction':
    st.markdown("<h2 class='title'>Parkinson's Disease Prediction Using ML</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    fo = col1.slider('MDVP:Fo(Hz) - Average vocal fundamental frequency', 50.0, 260.0, 120.0)
    fhi = col2.slider('MDVP:Fhi(Hz) - Maximum vocal fundamental frequency', 100.0, 600.0, 200.0)
    flo = col3.slider('MDVP:Flo(Hz) - Minimum vocal fundamental frequency', 50.0, 260.0, 110.0)
    jitter_percent = col1.slider('MDVP:Jitter(%)', 0.0, 1.0, 0.01)
    jitter_abs = col2.slider('MDVP:Jitter(Abs)', 0.0, 0.1, 0.0001)
    rap = col3.slider('MDVP:RAP', 0.0, 1.0, 0.01)
    ppq = col1.slider('MDVP:PPQ', 0.0, 1.0, 0.01)
    ddp = col2.slider('Jitter:DDP', 0.0, 1.0, 0.02)
    shimmer = col3.slider('MDVP:Shimmer', 0.0, 1.0, 0.05)
    shimmer_db = col1.slider('MDVP:Shimmer(dB)', 0.0, 10.0, 0.5)
    apq3 = col2.slider('Shimmer:APQ3', 0.0, 1.0, 0.02)
    apq5 = col3.slider('Shimmer:APQ5', 0.0, 1.0, 0.03)
    apq = col1.slider('MDVP:APQ', 0.0, 1.0, 0.03)
    dda = col2.slider('Shimmer:DDA', 0.0, 1.0, 0.05)
    nhr = col3.slider('NHR', 0.0, 1.0, 0.02)
    hnr = col1.slider('HNR', 0.0, 40.0, 20.0)
    rpde = col2.slider('RPDE', 0.0, 1.0, 0.5)
    dfa = col3.slider('DFA', 0.0, 1.0, 0.75)
    spread1 = col1.slider('spread1', -10.0, 0.0, -4.0)
    spread2 = col2.slider('spread2', 0.0, 1.0, 0.3)
    d2 = col3.slider('D2', 0.0, 3.0, 2.0)
    ppe = col1.slider('PPE', 0.0, 1.0, 0.1)

    user_input = [fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db,
                  apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]

    if st.button("Get Parkinson's Test Result"):
        pred = parkinsons_model.predict([user_input])[0]
        diagnosis = 'The person is likely to have Parkinson\'s disease 🧠' if pred == 1 else 'The person is not likely to have Parkinson\'s disease 🌟'
        st.markdown(f"<p class='result-text'>{diagnosis}</p>", unsafe_allow_html=True)
