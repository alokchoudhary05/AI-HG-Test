import streamlit as st
from streamlit_option_menu import option_menu
import pickle
import joblib
import warnings
import pandas as pd
import plotly.express as px
from io import StringIO
import requests

from Symptoms import Symptoms_Disease_Prediction
from codebase.dashboard_graphs import PregnancyDashboard
from dashboard import AIHEALTHGUARDDASHBOARD

pregnancy_model = joblib.load(open("model/Pregnancy.pkl",'rb'))
diabetic_model = pickle.load(open("model/Diabetes.sav",'rb'))
heart_model = pickle.load(open("model/Heart.sav",'rb'))

# sidebar for navigation
with st.sidebar:
    st.title("AI Health Guard")
    st.write("Welcome to the AI Health Guard")
    st.write("Choose an option from the menu below to get started:")

    selected = option_menu('AI Health Guard',
                          
                          ['About us',
                           'Disease Prediction and Recommendations',
                            'Pregnancy Risk Prediction',
                           'Diabetes Prediction',
                           'Heart Disease Prediction',
                           'Pregnancy Dashboard',
                           'AI-Health Guard Dashboard',
                           ],
                          icons=['chat-square-text','hospital','capsule-pill','heart','clipboard-data','chat-square'],
                          default_index=0)



if (selected == 'About us'):
    

    # Section 1: Welcome
    st.title("Welcome to AI Health Guard")
    st.subheader("Your Personalized Health Advisor. Predicts diseases, offers tailored medical advice, workouts, and diet plans for holisticwell-being.")
    st.write("At AI Health Guard, our mission is to revolutionize healthcare by offering innovative solutions through predictive analysis. "
         "Our platform is specifically designed to address the intricate aspects of Pregnancy Risk Prediction, Diabetes Prediction, Heart Disease Prediction, providing accurate "
         "predictions and proactive risk management.")
    
    st.image("graphics/fetal_health_image.jpg", use_column_width=True)
    

    # Section 2: Motivation
    st.header("Motivation")
    st.write("In remote and rural areas with few medical professionals, healthcare becomes a serious "
            " problem. The AI Health Guard Project is a new beacon for such regions. Applying Data "
            " Science and Machine Learning (ML) technologies, it seeks to link the far reaches of "
            " health services with neglected population groups.")
    st.write("The AI Health Guard project is a new frontier in healthcare provision. Its focus is on quick "
            " diagnosis, customized health counseling and preventative advice in a part of the world "
            " where there are few doctors to turn to. With the superior algorithms and creative "
            " platform technologies of the project, it's hoped to enable individuals to achieve wellness proactively rather than passively.")
    st.write(" Specifically, the AI Health Guard project’s design element is to address the needs of rural "
            " communities, where healthcare professionals’ shortage typically results in late diagnosis "
            " and treatment. This project, being built upon the capabilities of data science and ML, "
            " aims to deliver a solution that not only helps identify the disease but also make "
            " actionable suggestions and recommendations based on the user’s data. ")
    st.write("Overall, AI Health Guard is set to be a groundbreaking addition to the field of health "
            " technology, offering a proactive and personalized approach to health maintenance and disease prevention.")


    st.image("graphics/pregnancy_risk_image.jpg", use_column_width=True)


    # # Section 3: Dashboard
    # st.header("Dashboard")
    # st.write("Our Dashboard provides a user-friendly interface for monitoring and managing health data. It offers a holistic "
    #         "view of predictive analyses, allowing healthcare professionals and users to make informed decisions. The Dashboard "
    #         "is designed for ease of use and accessibility.")


    # Section 4: Conclusion
    st.header("Conclusion")
    st.write("The AI Health Guard project represents a significant endeavor in utilizing data science "
            " and machine learning techniques to empower individuals in managing their health "
            " effectively. By leveraging advanced algorithms and personalized recommendations, the " 
            " system aims to enhance healthcare outcomes and promote overall well-being.")
    # Closing note
    st.write("Thank you for choosing AI Health Guard. We are committed to advancing healthcare through technology and predictive analytics. "
            "Feel free to explore our features and take advantage of the insights we provide.")



# Symptoms Based Disease Predictions and Recommendations
if (selected == "Disease Prediction and Recommendations"):
    model = Symptoms_Disease_Prediction()
    model.main()



# Pregnancy Risk Prediction Page
if (selected == 'Pregnancy Risk Prediction'):
    
    # page title
    st.title('Pregnancy Risk Prediction')
    content = "Predicting the risk in pregnancy involves analyzing several parameters, including age, blood sugar levels, blood pressure, and other relevant factors. By evaluating these parameters, we can assess potential risks and make informed predictions regarding the pregnancy's health"
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age of the Person', key = "age")
        
    with col2:
        diastolicBP = st.text_input('diastolicBP in mmHg')
    
    with col3:
        BS = st.text_input('Blood glucose in mmol/L')
    
    with col1:
        bodyTemp = st.text_input('Body Temperature in Celsius')

    with col2:
        heartRate = st.text_input('Heart rate in beats per minute')
    
    riskLevel=""
    predicted_risk = [0] 
    # creating a button for Prediction
    with col1:
        if st.button('Pregnancy Test Result'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                predicted_risk = pregnancy_model.predict([[age, diastolicBP, BS, bodyTemp, heartRate]])
            # st
            st.subheader("Risk Level:")
            if predicted_risk[0] == 0:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: green;">Low Risk</p></bold>', unsafe_allow_html=True)
            elif predicted_risk[0] == 1:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: orange;">Medium Risk</p></Bold>', unsafe_allow_html=True)
            else:
                st.markdown('<bold><p style="font-weight: bold; font-size: 20px; color: red;">High Risk</p><bold>', unsafe_allow_html=True)
    with col2:
        if st.button("Clear"): 
            st.rerun()




# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    
    # page title
    st.title('Diabetes Prediction')
    # page title
    content = "Diabetes prediction involves using medical data and advanced algorithms to estimate the likelihood of an individual developing diabetes. This can include analyzing factors such as age, BMI, blood pressure, and family history. Early prediction aids in timely intervention and management, potentially preventing the onset of the disease."
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)
    
    
    # getting the input data from the user
    col1, col2, col3 = st.columns(3)
    
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
        
    with col2:
        Glucose = st.text_input('Glucose Level')
    
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    
    with col2:
        Insulin = st.text_input('Insulin Level')
    
    with col3:
        BMI = st.text_input('BMI value')
    
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    
    with col2:
        Age = st.text_input('Age of the Person')
    
    
    # code for Prediction
    diab_diagnosis = ""
    
    # creating a button for Prediction
    with col1:
        if st.button('Diabetes Test Result'):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
            diab_prediction = diabetic_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        
            if (diab_prediction[0] == 1):
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
    
    with col2:
        if st.button("Clear"): 
            st.rerun()
    st.success(diab_diagnosis)




# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    
    # Page title
    st.title('Heart Disease Prediction')

    content = ("Heart disease prediction involves using medical data and machine learning algorithms to identify "
               "individuals at risk of cardiovascular conditions. Accurate prediction models can help in early diagnosis "
               "and personalized treatment plans. Key factors often include age, lifestyle habits, genetic predisposition, "
               "and medical history.")
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.selectbox('Sex', ('Select','Male', 'Female'))
        
    with col3:
        cp = st.selectbox('Chest Pain types', ('Select','Low pain', 'Mild pain', 'Moderate pain', 'Extreme pain'))
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholesterol in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by fluoroscopy')
        

    with col1:
        st.markdown("<style>div[class*='stSelectbox'] > div[role='combobox'] { width: 200%; }</style>", unsafe_allow_html=True)
        thal = st.selectbox('Thalassemia', ('Select',
                                            'Normal (No Thalassemia)', 
                                            'Fixed Defect (Beta-thalassemia minor)', 
                                            'Reversible Defect (Beta-thalassemia intermedia)', 
                                            'Serious Defect (Beta-thalassemia major)'))

    # Convert input values to the required format
    sex = 0 if sex == 'Male' else 1
    cp_dict = {
        'Select':0,
        'Low pain': 0,
        'Mild pain': 1,
        'Moderate pain': 2,
        'Extreme pain': 3
    }
    cp = cp_dict[cp]

    thal_dict = {
        'Select':0,
        'Normal (No Thalassemia)': 0,
        'Fixed Defect (Beta-thalassemia minor)': 1,
        'Reversible Defect (Beta-thalassemia intermedia)': 2,
        'Serious Defect (Beta-thalassemia major)': 3
    }
    thal = thal_dict[thal]

    # Code for Prediction
    heart_diagnosis = ""
    col1, col2 = st.columns(2)

    # Creating a button for Prediction
    with col1:
        if st.button('Heart Disease Test Result'):
            try:
                input_data = [[int(age), sex, cp, int(trestbps), int(chol), int(fbs), int(restecg), int(thalach), 
                               int(exang), float(oldpeak), int(slope), int(ca), thal]]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    heart_prediction = heart_model.predict(input_data)
                
                if heart_prediction[0] == 1:
                    heart_diagnosis = 'The person is having heart disease'
                else:
                    heart_diagnosis = 'The person does not have any heart disease'
                
                st.success(heart_diagnosis)
            except Exception as e:
                st.error(f"Error in prediction: {e}")

    with col2:
        if st.button("Clear"): 
            st.experimental_rerun()




# Dashboard analysis Page
if (selected == "Pregnancy Dashboard"):
    api_key = "579b464db66ec23bdd00000139b0d95a6ee4441c5f37eeae13f3a0b2"
    api_endpoint = api_endpoint= f"https://api.data.gov.in/resource/6d6a373a-4529-43e0-9cff-f39aa8aa5957?api-key={api_key}&format=csv"
    st.header("Dashboard")
    content = "Our interactive dashboard offers a comprehensive visual representation of maternal health achievements across diverse regions. The featured chart provides insights into the performance of each region concerning institutional deliveries compared to their assessed needs. It serves as a dynamic tool for assessing healthcare effectiveness, allowing users to quickly gauge the success of maternal health initiatives."
    st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div></br>", unsafe_allow_html=True)

    dashboard = PregnancyDashboard(api_endpoint)
    dashboard.create_bubble_chart()
    with st.expander("Show More"):
    # Display a portion of the data
        content = dashboard.get_bubble_chart_data()
        st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)

    dashboard.create_pie_chart()
    with st.expander("Show More"):
    # Display a portion of the data
        content = dashboard.get_pie_graph_data()
        st.markdown(f"<div style='white-space: pre-wrap;'><b>{content}</b></div>", unsafe_allow_html=True)




# AI-Health Guard Dashboard analysis Page
if (selected == "AI-Health Guard Dashboard"):
    dashboard = AIHEALTHGUARDDASHBOARD()
    dashboard.main()