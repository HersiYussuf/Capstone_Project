# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# from PIL import Image

# header = st.container()
# dataset = st.container()
# features = st.container()
# model_training = st.container()

# with header:
#     st.title('Welcome to the Diabetes readmission Prediction App')
#     st.text('This app predicts the possible readmission of a patient based on the below features')
    
    
# with dataset:
#     st.header('Patient dataset')
#     st.text('The data is taken from the UCI diabetes dataset')
#     st.text('The data is split into training and testing sets')
#     st.text('The training set is used to train the model')
#     st.text('The testing set is used to test the model')
#     st.text('The model is trained on the training set')
#     st.text('The model is tested on the testing set')
#     st.text('The model is then saved in the app')
#     st.text('The model is then used to predict the readmission of a patient')
    
    
    
# with features:
#     st.header('Features')
#     st.text('The features used to predict the readmission are:')
#     st.text('race')
#     st.text('gender')
#     st.text('age')
#     st.text('discharge_disposition_id')
#     st.text('admission_source_id')
#     st.text('time_in_hospital')
#     st.text('num_lab_procedures')
#     st.text('num_procedures')
#     st.text('num_medications')
#     st.text('diag_1')
#     st.text('diag_2')
#     st.text('diag_3')
#     st.text('number_diagnoses')
#     st.text('max_glu_serum')
#     st.text('A1Cresult')
#     st.text('metformin')
#     st.text('repaglinide')
#     st.text('glimepiride')
#     st.text('pioglitazone')
#     st.text('glipizide')
#     st.text('glyburide')


# with model_training:
#     st.header('Model training')
#     st.text('The model is trained on the training set')
#     st.text('The model is tested on the testing set')
#     st.text('The model is then saved in the app')
#     st.text('The model is then used to predict the readmission of a patient')
    
    
    
# # Load the pickled model file
# with open('xgb_model.pkl', 'rb') as f:
#     model = pickle.load(f)

# # Create a button for prediction
# prediction_button = st.button('Predict')

# # Check if the button is clicked
# if prediction_button:
#     # Perform prediction using the loaded model
#     # You can write the necessary code here to take user inputs and make predictions
#     # Make sure the input features match the requirements of the trained model
#     # Display the prediction result using st.write or any other suitable Streamlit function
#     prediction = model.predict(...)
#     st.write('The predicted readmission:', prediction)
################################################################################################
######################################################################
############################################################################################
import streamlit as st
import pickle

# Set page configurations
st.set_page_config(page_title='Diabetes Readmission Prediction App')

# Home page
st.title('Diabetes Readmission Prediction App')
st.markdown('This app predicts the possible readmission of a patient based on various features.')
st.markdown('Please select an option from the sidebar.')

# Sidebar navigation
menu_selection = st.sidebar.selectbox('Menu', ('Home', 'Dataset', 'Features', 'Model Training'))

# Dataset page
if menu_selection == 'Dataset':
    st.header('Patient Dataset')
    st.markdown('The data is taken from the UCI diabetes dataset.')
    st.markdown('The data is split into training and testing sets.')
    st.markdown('The training set is used to train the model.')
    st.markdown('The testing set is used to test the model.')
    st.markdown('The model is trained on the training set.')
    st.markdown('The model is tested on the testing set.')
    st.markdown('The model is then saved in the app.')
    st.markdown('The model is then used to predict the readmission of a patient.')

# Features page
elif menu_selection == 'Features':
    st.header('Features')
    with st.container():
        st.markdown('The features used to predict the readmission are:')
        features = ['- race', '- gender', '- age', '- discharge_disposition_id', '- admission_source_id',
                    '- time_in_hospital', '- num_lab_procedures', '- num_procedures', '- num_medications',
                    '- diag_1', '- diag_2', '- diag_3', '- number_diagnoses', '- max_glu_serum',
                    '- A1Cresult', '- metformin', '- repaglinide', '- glimepiride', '- pioglitazone',
                    '- glipizide', '- glyburide']
        for feature in features:
            st.markdown(feature)

# Model Training page
elif menu_selection == 'Model Training':
    st.header('Model Training')
    st.markdown('The model is trained on the training set.')
    st.markdown('The model is tested on the testing set.')
    st.markdown('The model is then saved in the app.')
    st.markdown('The model is then used to predict the readmission of a patient.')

# Main page
else:
    st.header('Welcome to the Diabetes Readmission Prediction App')
    st.markdown('This app allows you to predict the possible readmission of a patient based on various features.')
    st.markdown('Please use the sidebar to navigate through different sections.')

# Load the pickled model file
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a button for prediction
prediction_button = st.button('Predict')

# Check if the button is clicked
if prediction_button:
    # Perform prediction using the loaded model
    # You can write the necessary code here to take user inputs and make predictions
    # Make sure the input features match the requirements of the trained model
    # Display the prediction result using st.write or any other suitable Streamlit function
    prediction = model.predict(...)
    st.write('The predicted readmission:', prediction)
