import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('random_forest_model.pickle', 'rb') as model_file:
    model = pickle.load(model_file)

# Define the Streamlit app
def main():
    st.title('Diabetes Prediction App')
    st.sidebar.header('User Input Features')

    # Collect user input features
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 23)
    insulin = st.sidebar.slider('Insulin', 0, 846, 30)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    diabetes_pedigree_function = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Age', 21, 81, 29)

    # Create a button for prediction
    if st.button('Predict'):
        # Create feature array for prediction
        features = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age])

        # Make prediction
        prediction = model.predict([features])[0]

        st.write('## Prediction:')
        if prediction == 1:
            st.write('**Diabetic**')
        else:
            st.write('**Non-Diabetic**')

# Run the app
if __name__ == '__main__':
    main()
