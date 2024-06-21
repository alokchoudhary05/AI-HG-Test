import streamlit as st
import numpy as np
import pandas as pd
import pickle


# Train data loading 
@st.cache_data
def load_data(file_path):
    df_v1 = pd.read_csv(file_path)
    return df_v1


class Symptoms_Disease_Prediction:
    # Main function to render the dashboard
    @staticmethod
    def main():

        # Load database datasets
        sym_des = pd.read_csv("datasets/symtoms_df.csv")
        precautions = pd.read_csv("datasets/precautions_df.csv")
        workout = pd.read_csv("datasets/workout_df.csv")
        description = pd.read_csv("datasets/description.csv")
        medications = pd.read_csv('datasets/medications.csv')
        diets = pd.read_csv("datasets/diets.csv")

        # Load model
        svc = pickle.load(open('datasets/svc.pkl', 'rb'))
        le = pickle.load(open('datasets/label_encoder.pkl', 'rb'))

        # Normalize column names and data to handle inconsistencies
        workout.rename(columns={'disease': 'Disease'}, inplace=True)

        def normalize_column(df, column_name):
            df[column_name] = df[column_name].str.strip().str.lower()

        for df in [description, precautions, medications, workout, diets]:
            normalize_column(df, 'Disease')

        # Function to predict disease based on symptoms
        def predict_disease(symptoms):
            symptoms_dict = {symptom: 0 for symptom in svc.feature_names_in_}
            for symptom in symptoms:
                symptom = symptom.strip().lower()
                if symptom in symptoms_dict:
                    symptoms_dict[symptom] = 1

            input_data = pd.DataFrame([symptoms_dict])
            predicted_disease = svc.predict(input_data)
            disease_name = le.inverse_transform(predicted_disease)[0].strip().lower()

            return disease_name

        # Helper function to fetch recommendations
        def helper(dis):
            desc = description[description['Disease'] == dis]['Description'].values[0]
            pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values[0]
            med = medications[medications['Disease'] == dis]['Medication'].values[0].split(',')
            die = diets[diets['Disease'] == dis]['Diet'].values[0].split(',')
            wrkout = workout[workout['Disease'] == dis]['workout'].values[0].split(',')

            return desc, pre, med, die, wrkout
        
        # Define the dataset path
        data_file_path_v1 = 'Dataset/df_v1.csv'
        df_v1 = load_data(data_file_path_v1)
        
        # Streamlit app
        st.title('Disease Prediction and Recommendations')

        # symptoms_input = st.text_input('Enter symptoms separated by commas (e.g. headache, fever)')
        symptoms_input = st.multiselect("Enter Symptoms ", df_v1.columns)

        if st.button('Predict'):
            if not symptoms_input:
                st.error('Please enter symptoms')
            else:
                user_symptoms = [s.strip().lower() for s in symptoms_input]
                predicted_disease = predict_disease(user_symptoms)
                dis_des, precautions_list, medications, diet, workout = helper(predicted_disease)

                st.write(f"**Predicted Disease:** {predicted_disease.capitalize()}")
                st.write(f"**Description:** {dis_des}")
                st.write("**Precautions:**")
                for precaution in precautions_list:
                    st.write(f"- {precaution}")
                st.write("**Medications:**")
                for med in medications:
                    st.write(f"- {med}")
                st.write("**Diet:**")
                for die in diet:
                    st.write(f"- {die}")
                st.write("**Workout:**")
                for wrk in workout:
                    st.write(f"- {wrk}")

if __name__ == "__main__":
    Symptoms_Disease_Prediction.main()