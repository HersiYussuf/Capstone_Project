import os
import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb
from PIL.Image import Image
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# Specify the directory path
directory = r'C:\Users\wanji\OneDrive\Desktop\Diabetic Admissio.py'

# Create the directory if it does not exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Load the clean data set
df = pd.read_csv(r"C:\Users\wanji\Desktop\clean_data.csv")

# Separate features and target
X = df.drop('readmitted', axis=1)
y = df['readmitted']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the scalers
numeric_features = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures',
                    'num_medications', 'number_diagnoses', 'num_total_visits']
categorical_features = ['race', 'gender', 'discharge_disposition_id', 'admission_source_id']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Apply transformations to the features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])


# Fit and transform the preprocessor on the training data
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

# XGBoost
# XGBoost
xgb_model_final = make_pipeline(
    xgb.XGBClassifier(random_state=42)
)

xgb_model_final.fit(X_train_scaled, y_train)
xgb_model_final.steps[0][1].save_model(os.path.join(directory, 'xgb_model_final.model'))

# Save the preprocessor
joblib.dump(preprocessor, os.path.join(directory, 'preprocessor.joblib'))

# Load trained model
xgb_model_final = xgb.XGBClassifier()
xgb_model_final.load_model(os.path.join(directory, 'xgb_model_final.model'))

# Load preprocessor
preprocessor = joblib.load(os.path.join(directory, 'preprocessor.joblib'))

# Mapping for non-integer values in specific fields
discharge_disposition_mapping = {
    'Discharged to home': 1,
    # Add more mappings as needed
}

admission_source_mapping = {
    'Transferred from another health care facility': 1,
    'Referral': 2,
    'Not Available': 3,
    'Emergency': 4,
    # Add more mappings as needed
}

class_mapping = {
    'readmission': 1,
    'no readmission': 0
}

admission_source_mapping_inverse = {v: k for k, v in admission_source_mapping.items()}

max_glu_serum_mapping_source = {
    'None': 1,
    'Norm': 2,
    '>300': 3,
    '>200': 4,
    # Add more mappings as needed
}

feature_names = [
    'race', 'gender', 'age', 'discharge_disposition_id',
    'admission_source_id', 'time_in_hospital', 'num_lab_procedures',
    'num_procedures', 'num_medications', 'diag_1', 'diag_2', 'diag_3',
    'number_diagnoses', 'max_glu_serum', 'A1Cresult', 'metformin',
    'repaglinide', 'glimepiride', 'glipizide', 'glyburide', 'pioglitazone',
    'rosiglitazone', 'insulin', 'change', 'diabetesMed',
    'num_total_visits'
]

def main():

    # Initialize session state
    if 'biodata' not in st.session_state:
        st.session_state['biodata'] = {}  # Initialize with an empty dictionary

    # Initialize session state
    if 'medical_procedures' not in st.session_state:
        st.session_state['medical_procedures'] = {}  # Initialize with an empty dictionary

    # Initialize with an empty dictionary
    if 'medicine' not in st.session_state:
        st.session_state['medicine'] = {}  # Initialize with an empty dictionary

    # Initialize with an empty dictionary
    if 'result' not in st.session_state:
        st.session_state['result'] = {}  # Initialize with an empty dictionary

    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 0

    pages = [
        biodata_page,
        medical_procedures_page,
        medicine_page,
        result_page
    ]

    current_page = st.session_state.get('current_page', 0)


    if current_page < len(pages):
        pages[current_page]()

        if current_page > 0:
            col1, col2 = st.columns([1, 3])
            if col1.button('Previous'):
                st.session_state['current_page'] -= 1

        if current_page < len(pages) - 1:
            col1, col2 = st.columns([3, 1])
            if col2.button('Next'):
                st.session_state['current_page'] += 1
    else:
        st.session_state.pop('current_page', None)

def biodata_page():
    st.header('BIO-DATA')

    age = st.text_input('Age')
    discharge_disposition_id = st.selectbox('Discharge Disposition ID', list(discharge_disposition_mapping.keys()))
    admission_source_id = st.selectbox('Admission Source ID', list(admission_source_mapping.keys()))
    time_in_hospital = st.text_input('Time In Hospital')
    num_total_visits = st.text_input('Number of Total Visits')
    race = st.selectbox('Race', ['Caucasian', 'AfricanAmerican', 'Asian', 'Other', 'Hispanic'])
    gender = st.selectbox('Gender', ['Female', 'Male'])

    if st.button('Next', key='button1'):
        st.session_state['current_page'] += 1

    data = {
        'age': age,
        'discharge_disposition_id': discharge_disposition_id,
        'admission_source_id': admission_source_id,
        'time_in_hospital': time_in_hospital,
        'num_total_visits': num_total_visits,
        'race': race,
        'gender': gender
    }
    st.session_state['biodata'] = data

def medical_procedures_page():
    st.header('MEDICAL PROCEDURES')

    num_lab_procedures = st.text_input('Num Lab Procedures')
    num_procedures = st.text_input('Num Procedures')
    num_medications = st.text_input('Num Medications')
    number_diagnoses = st.text_input('Number of Diagnoses')
    diag_1 = st.selectbox('Diag 1',['Circulatory','Diabetes','Endocrine, Nutritional, Metabolic, Immunity','Respiratory','Genitourinary','External causes of injury','Digestive','Mental Disorders','Skin and Subcutaneous Tissue','Blood and Blood-Forming Organs','Other Symptoms','Musculoskeletal System and Connective Tissue','Injury and Poisoning','Infectious and Parasitic','Neoplasms','Nervous','Congenital Anomalies','Pregnancy, Childbirth','Sense Organs'])
    diag_2 = st.selectbox('Diag 2',['Circulatory','Diabetes','Endocrine, Nutritional, Metabolic, Immunity','Respiratory','Genitourinary','External causes of injury','Digestive','Mental Disorders','Skin and Subcutaneous Tissue','Blood and Blood-Forming Organs','Other Symptoms','Musculoskeletal System and Connective Tissue','Injury and Poisoning','Infectious and Parasitic','Neoplasms','Nervous','Congenital Anomalies','Pregnancy, Childbirth','Sense Organs'])
    diag_3 = st.selectbox('Diag 3',['Circulatory','Diabetes','Endocrine, Nutritional, Metabolic, Immunity','Respiratory','Genitourinary','External causes of injury','Digestive','Mental Disorders','Skin and Subcutaneous Tissue','Blood and Blood-Forming Organs','Other Symptoms','Musculoskeletal System and Connective Tissue','Injury and Poisoning','Infectious and Parasitic','Neoplasms','Nervous','Congenital Anomalies','Pregnancy, Childbirth','Sense Organs'])

    if st.button('Next', key='button2'):
        st.session_state['current_page'] += 1

    data = {
        'num_lab_procedures': num_lab_procedures,
        'num_procedures': num_procedures,
        'num_medications': num_medications,
        'number_diagnoses': number_diagnoses,
        'diag_1': diag_1,
        'diag_2': diag_2,
        'diag_3': diag_3,
    }
    st.session_state['medical_procedures'] = data


def medicine_page():
    st.header('MEDICINE')

    max_glu_serum = st.selectbox('Max Glu Serum', ['None', 'Norm', '>300', '>200'])
    metformin = st.selectbox('Metformin', ['No', 'Down', 'Steady', 'Up'])
    repaglinide = st.selectbox('Repaglinide', ['No', 'Down', 'Steady', 'Up'])
    glimepiride = st.selectbox('Glimepiride', ['No', 'Down', 'Steady', 'Up'])
    glipizide = st.selectbox('Glipizide', ['No', 'Down', 'Steady', 'Up'])
    glyburide = st.selectbox('Glyburide', ['No', 'Down', 'Steady', 'Up'])
    pioglitazone = st.selectbox('Pioglitazone', ['No', 'Down', 'Steady', 'Up'])
    rosiglitazone = st.selectbox('Rosiglitazone', ['No', 'Down', 'Steady', 'Up'])
    insulin = st.selectbox('Insulin', ['No', 'Down', 'Steady', 'Up'])

    if st.button('Next', key='button3'):
        st.session_state['current_page'] += 1

    data = {
        'max_glu_serum': max_glu_serum,
        'metformin': metformin,
        'repaglinide': repaglinide,
        'glimepiride': glimepiride,
        'glipizide': glipizide,
        'glyburide': glyburide,
        'pioglitazone': pioglitazone,
        'rosiglitazone': rosiglitazone,
        'insulin': insulin
        # Add this line
    }
    st.session_state['medicine'] = data


def result_page():
    st.header('RESULTS')

    A1Cresult = st.selectbox('A1C Result', ['None', 'Norm', '>7', '>8'])
    change = st.selectbox('Change', ['0', '1'])
    diabetesMed = st.selectbox('Diabetes Med', ['0', '1'])

    if st.button('Predict', key='button4'):
        xgb_prediction = predict()
        st.session_state['xgb_prediction'] = xgb_prediction

        st.session_state['current_page'] += 1

    data = {
        'A1Cresult': A1Cresult,
        'change': change,
        'diabetesMed': diabetesMed
    }
    st.session_state['result'] = data

    if 'xgb_prediction' in st.session_state:
        xgb_prediction = st.session_state['xgb_prediction']
        st.subheader('Result')
        st.write('Prediction:', xgb_prediction)


import numpy as np


def replace_unknown_and_nan(value):
    if pd.isnull(value) or value == 'Unknown':
        return 9999
    return value


def predict():
    biodata = st.session_state['biodata']
    medical_procedures = st.session_state['medical_procedures']
    medicine = st.session_state['medicine']
    result = st.session_state['result']

    # Data
    data = {**biodata, **medical_procedures, **medicine, **result}

    # Replace missing values and empty strings with a default value
    data = {k: v if pd.notna(v) and v != '' else 'Unknown' for k, v in data.items()}

    # Mapping for non-integer values in specific fields
    discharge_disposition_mapping_inverse = {v: k for k, v in discharge_disposition_mapping.items()}
    admission_source_mapping_inverse = {v: k for k, v in admission_source_mapping.items()}
    max_glu_serum_mapping_inverse = {v: k for k, v in max_glu_serum_mapping_source.items()}

    # Convert non-integer values to their numeric representations
    data['discharge_disposition_id'] = discharge_disposition_mapping_inverse.get(data['discharge_disposition_id'],
                                                                                 'Unknown')

    if 'admission_source_id' in data:
        data['admission_source_id'] = admission_source_mapping_inverse.get(data['admission_source_id'], 'Unknown')

    data['max_glu_serum'] = max_glu_serum_mapping_inverse.get(data['max_glu_serum'], 'Unknown')

    # Encode categorical features using OrdinalEncoder
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    categorical_features_to_encode = ['race', 'gender', 'admission_source_id', 'max_glu_serum', 'A1Cresult',
                                      'diabetesMed']

    for feature in categorical_features_to_encode:
        if feature in data and data[feature] != 'Unknown':
            data[feature] = encoder.fit_transform([[data[feature]]])[0][0]

    # Create a DataFrame from the data
    data_df = pd.DataFrame([data], columns=feature_names)

    # Replace missing values with a placeholder value
    data_df_filled = data_df.fillna(value=-9999).astype(str)

    # Apply one-hot encoding to non-numeric columns
    non_numeric_columns = data_df_filled.select_dtypes(exclude=np.number).columns
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoded_data = encoder.fit_transform(data_df_filled[non_numeric_columns])

    # Combine encoded data with numeric columns
    numeric_columns = data_df_filled.select_dtypes(include=np.number).columns
    data_preprocessed = np.concatenate((data_df_filled[numeric_columns].values, encoded_data), axis=1)

    # Drop missing values
    data_df = data_df.dropna()

    if data_df.empty:
        return 'Invalid Input'

    # Check for NaN values using a custom function

    def check_nan(x):
        if pd.isnull(x):
            return False
        elif isinstance(x, str) and x == 'Unknown':
            return False
        else:
            return True

    data_df = data_df.applymap(check_nan)

    # Apply the preprocessor transformation
    # Apply the preprocessor transformation
    try:
        data_preprocessed = preprocessor.transform(data_df_filled.astype(str))

    except ValueError:
        return 'Invalid Input'

    # Make predictions using the XGBoost model
    xgb_prediction = xgb_model_final.predict_proba(data_preprocessed)
    prob_no_readmission = xgb_prediction[0][class_mapping['no readmission']]
    prob_readmission = xgb_prediction[0][class_mapping['readmission']]

    return f"Probability of No Readmission: {prob_no_readmission:.2f}, Probability of Readmission: {prob_readmission:.2f}"

if __name__ == '__main__':
    main()
