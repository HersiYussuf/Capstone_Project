![Alt text](output.png)
# <span style="color:blue">HEALTHCARE DATA ANALYSIS: PREDICTING HOSPITAL READMISSION RATES.</SPAN>

## Project Description

Diabetes is a chronic disease characterized by an extended level of blood glucose in the body. It is influenced by factors such as height, race, gender, age, and sugar concentration. Early hospital readmissions in patients with diabetes pose significant challenges and impact healthcare quality.

In this project, we aim to leverage machine learning techniques to analyze a large clinical database and understand historical patterns of diabetes care in patients admitted to a US hospital. Our objective is to identify insights that can contribute to improvements in patient safety and healthcare outcomes.

By utilizing machine learning algorithms, we will develop predictive models to determine the likelihood of early hospital readmission for patients with diabetes. These models will assist healthcare professionals in intervening and providing timely and effective care to prevent readmissions.

The project will focus on data analysis, preprocessing, feature engineering, and model development. We will evaluate the performance of the models using appropriate evaluation metrics and present the findings.

Through this project, we aim to showcase how machine learning can be utilized to address the challenges caused by early hospital readmissions in diabetic patients, ultimately improving healthcare quality and patient outcomes.



## Business Problem and Constraints
## Business Problem

The business problem we aim to address is the high rate of readmissions among patients with diabetes in the United States.

**Background:** It is estimated that 9.3% of the population in the United States has diabetes, with 28% of cases going undiagnosed. The readmission rates for diabetic patients are alarmingly high, with a 30-day readmission rate ranging from 14.4% to 22.7%. The rates of readmissions beyond 30 days are even higher, with over 26% of diabetic patients being readmitted within 3 months and 30% within 1 year. These frequent readmissions result in significant healthcare costs, with an estimated $124 billion associated with the hospitalization of diabetic patients in the USA. Of this, $25 billion is attributable to 30-day readmissions assuming a 20% readmission rate.

## Constraints

1. **Interpretability of Model**: In the healthcare domain, interpretability of the model is crucial. It is essential to understand why the model predicts a patient's readmission to provide clear explanations to healthcare professionals and patients.

2. **Latency is not Strictly Important**: Most healthcare-related applications are not latency-dependent. While timely predictions are valuable, the focus is primarily on accuracy and interpretability rather than real-time response.

3. **High Cost of Misclassification**: Misclassification of readmission decisions can have significant financial implications. Incorrectly predicting readmission for patients who don't require it can put a financial burden on them. Conversely, failing to predict readmission for patients who need it can lead to increased readmission costs for hospitals. Therefore, minimizing the misclassification rate is critical.



## Data Overview
## Data Overview

The dataset used for this project is sourced from the Health Facts database, a national data warehouse maintained by Cerner Corporation in Kansas City, MO. This database collects comprehensive clinical records from hospitals across the United States.

**Dataset Filtering Criteria:** The dataset has been filtered based on the following criteria:

1. It includes inpatient encounters, which are hospital admissions.
2. It focuses on "diabetic" encounters, where any type of diabetes was diagnosed during the encounter.
3. The length of stay ranges from 1 day to 14 days.
4. Laboratory tests were performed during the encounter.
5. Medications were administered during the encounter.

By applying these criteria, approximately 100,000 records have been identified for further analysis.

The dataset comprises 55 features, including gender, weight, encounter ID, and readmission status. It can be utilized for classification and clustering tasks. The project goal is to predict whether a patient will be readmitted after treatment or not, which falls under a classification task.

**Feature Description:**

- Encounter ID: Unique identifier of an encounter
- Patient number: Unique identifier of a patient
- Race: Values include Caucasian, Asian, African American, Hispanic, and other
- Gender: Values include male, female, and unknown/invalid
- Age: Grouped in 10-year intervals (e.g., 0-10, 10-20, ..., 90-100)
- Weight: Weight in pounds
- Admission type: Integer identifier corresponding to 9 distinct values, such as emergency, urgent, elective, newborn, and not available
- Discharge disposition: Integer identifier corresponding to 29 distinct values, including discharged to home, expired, and not available
- Admission source: Integer identifier corresponding to 21 distinct values, such as physician referral, emergency room, and transfer from a hospital
- Time in hospital: Integer representing the number of days between admission and discharge
- Payer code: Integer identifier corresponding to 23 distinct values, such as Blue Cross/Blue Shield, Medicare, and self-pay Medical
- Medical specialty: Integer identifier corresponding to 84 distinct values, including cardiology, internal medicine, family/general practice, and surgeon
- Number of lab procedures: Number of lab tests performed during the encounter
- Number of procedures: Numeric value representing the number of procedures (other than lab tests) performed during the encounter
- Number of medications: Number of distinct generic names administered during the encounter
- Number of outpatient visits: Number of outpatient visits by the patient in the year preceding the encounter
- Number of emergency visits: Number of emergency visits by the patient in the year preceding the encounter
- Number of inpatient visits: Number of inpatient visits by the patient in the year preceding the encounter
- Diagnosis 1: The primary diagnosis coded as the first three digits of ICD9; includes 848 distinct values
- Diagnosis 2: Secondary diagnosis coded as the first three digits of ICD9; includes 923 distinct values
- Diagnosis 3: Additional secondary diagnosis coded as the first three digits of ICD9; includes 954 distinct values
- Number of diagnoses: Number of diagnoses entered into the system (0%)
- Glucose serum test result: Indicates the range of the result or if the test was not taken. Values include ">200," ">300," "normal," and "none" if not measured
- A1c test result: Indicates the range of the result or if the test was not taken. Values include ">8" (greater than 8%), ">7" (greater than 7% but less than 8%), "normal" (less than 7%), and "none" if not measured
- Change of medications: Indicates if there was a change in diabetic medications (dosage or generic name). Values include "change" and "no change"
- Diabetes medications: Indicates if any diabetic medication was prescribed. Values include "yes" and "no"
- 24 different kinds of medical drugs
- Readmitted: Days to inpatient readmission. Values include "❤0" (readmitted in less than 30 days), ">30" (readmitted in more than 30 days), and "No" for no record of readmission.



## ML Formulation



## Features


## Methodology



## Evaluation



## Results



## Dependencies



## Usage


## Contributing



## License



## Authors

- [Aaron Onserio](https://github.com/AaronOnserio)
- [Daniel Ekale](https://github.com/D-EKALE)
- [Emily Njue](https://github.com/EmillyN22)
- [Robert Mbau](https://github.com/robertmbau)
- [Yussuf Hersi](https://github.com/HersiYussuf)
- [Jimcollins Wamae](https://github.com/)
- [Edna Wanjiku](https://github.com/Edna722)

## Contact


