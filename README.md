![Alt text](images/output.png)
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



## **Data Overview**

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


## **Statistically Significant Features**
After conducting statistical analysis, we identified the following statistically significant features:

- age
- discharge_disposition_id
- admission_source_id
- time_in_hospital
- num_lab_procedures
- num_procedures
- num_medications
- diag_1
- diag_2
- diag_3
- number_diagnoses
- max_glu_serum
- A1Cresult
- metformin
- repaglinide
- glimepiride
- glipizide
- pioglitazone
- insulin
- change
- diabetesMed
- readmitted
- num_total_visits

These features were found to have statistical significance in relation to the dataset and are potentially important for further analysis and modeling.

Additionally, we observed that there were some medicines that were administered to a very small fraction of the total diabetic patients who visited the hospital (about 0.01 of the total population). To ensure the dataset's quality and avoid potential biases, we decided to drop these medicines if they are categorized as "statistically insignificant features" since they were dispensed to only a small number of patients.

Please refer to the the notebook for detailed information on the dataset and its features.



## **Modells used.**
#### <span style='color:cyan'>**1. Baseline model**</span>
A baseline model refers to a simple or basic model that serves as a starting point for comparison in machine learning or data analysis tasks. It is often the simplest model that can be built without any complex algorithms or feature engineering.

The purpose of a baseline model is to establish a benchmark or reference point against which the performance of more advanced or sophisticated models can be measured. It provides a point of comparison to assess whether the additional complexity and effort put into developing more advanced models yield significant improvements in performance.



#### <span style='color:cyan'>**2. Logistic regression: Imbalannced data**</span>
Logistic Regression with imbalanced data refers to using the same algorithm but without addressing the class imbalance issue. In this case, the model may face challenges in accurately predicting the minority class because it has fewer instances to learn from. The model's performance may be skewed towards the majority class, resulting in low recall (ability to identify positive instances) for the minority class.

Each of these models has its strengths and weaknesses, and the choice of which model to use depends on the specific problem and dataset at hand. It's important to evaluate and compare the performance of different models to select the most suitable one for the task.



#### <span style='color:orange'>**3. Logistic Regression: Balanced Data**</span>
Logistic Regression is a statistical model used for binary classification problems, where the goal is to predict one of two possible outcomes. It calculates the probability of an instance belonging to a particular class and makes a prediction based on that probability. Logistic Regression with balanced data refers to the use of a technique to address class imbalance, where the number of instances in one class is significantly higher than the other. By balancing the data, Logistic Regression can give equal importance to both classes and provide more accurate predictions.


#### <span style='color:orange'>**4. Random Forest**</span> 
Random Forest is a machine learning algorithm that combines multiple decision trees to make predictions. Think of it as a group of experts coming together to make a decision. Each decision tree in the Random Forest makes its prediction, and then the final prediction is determined by taking a majority vote among all the trees. This approach helps to reduce overfitting and improve the accuracy of the predictions. Random Forest is known for its ability to handle complex data and provide reliable results.
 #### <span style='color:orange'>**5. Decision Tree**</span>

A Decision Tree is a simple yet powerful algorithm that mimics the way humans make decisions. It starts with a single node called the root, which represents the entire dataset. The tree then splits the data based on different features, creating branches and sub-branches. Each branch represents a decision or outcome, leading to a final prediction or result at the leaf nodes. Decision Trees are easy to interpret and understand, making them useful for explaining the reasoning behind predictions.
 #### <span style='color:orange'>**6. XGBoost**</span> 
XGBoost stands for Extreme Gradient Boosting, and it is an advanced machine learning algorithm that is widely used for classification and regression tasks. XGBoost is similar to Random Forest in that it combines multiple models, but it has a different approach. It creates a series of decision trees, where each subsequent tree tries to correct the mistakes made by the previous trees. This iterative process continues until the model reaches an optimal state. XGBoost is known for its speed, scalability, and high performance in various machine learning competitions.# Model Performance Evaluation

The model performance evaluation included several algorithms such as Logistic Regression, Random Forest, Decision Tree, and XGBoost. The baseline model had poor performance, with low accuracy and recall. Logistic Regression on imbalanced data also performed poorly, indicating the impact of class imbalance. However, Logistic Regression on balanced data showed an improvement in recall but still fell short of expectations. Random Forest achieved the highest success metric with perfect recall for the positive class and high accuracy. Decision Tree had high accuracy but slightly lower recall. XGBoost performed well, with high recall and accuracy.

## Model Results

### Baseline Model

- Accuracy: 0.8882
- Precision: 0.88
- Recall: 0.00
- F1-score: 1.00
- Support: 21102

### Logistic Regression: Imbalanced Data

- Recall: 0.0046
- Precision: 0.2281
- F1-score: 0.0091
- Accuracy: 0.8812

### Logistic Regression: Balanced Data

- Recall: 0.05
- Precision: 0.61
- F1-score: 0.59
- Accuracy: 0.61

### Random Forest

- Precision: 0.94
- Recall: 1.00
- F1-score: 0.94
- Accuracy: 0.94

### Decision Tree

- Precision: 0.88
- Recall: 0.86
- F1-score: 0.88
- Accuracy: 0.86

### XGBoost

- Precision: 0.89
- Recall: 0.93
- F1-score: 0.93
- Accuracy: 0.93

## Conclusions

Based on the evaluation results, the following conclusions can be drawn:

- The baseline model performed poorly, with an accuracy of 8.882 and low recall for the positive class.
- Logistic Regression with imbalanced data had very low recall and precision.
- Logistic Regression with balanced data improved the recall to 0.05 but is still not satisfactory.
- Random Forest achieved the highest success metric with a recall of 1.00 and an accuracy of 0.94.
- Decision Tree had high accuracy but slightly lower recall compared to other models.
- XGBoost performed well with a recall of 0.93 and high accuracy.

## Recommendations

Based on the evaluation, the following recommendations can be made:

- Random Forest and XGBoost are recommended models due to their higher recall and accuracy compared to other models.
- Consider further optimizing the models to improve the recall for the positive class.
- Explore additional techniques to handle class imbalance, such as oversampling or undersampling.
- Gather more data, if possible, to increase the representation of the positive class and reduce class imbalance.
- Conduct feature engineering to identify more informative features that can enhance the model's predictive power.
- Explore ensemble methods to combine the strengths of multiple models and improve overall performance.
- Consider using a different evaluation metric, such as the F1-score, to evaluate the models.

## Challenges

The main challenges faced during the evaluation process include:
- Finding the right balance between model performance and computational efficiency.
- Dealing with class imbalance, where the majority class dominated the training data.
- Achieving a satisfactory recall for the positive class.
- Identifying the most informative features to improve the model's predictive power.
- Determining the best model to use for the classification task.
- Optimizing the model to achieve the best performance.
- Determining the best evaluation metric to use for the classification task.
- 

## Next Steps

To further improve the model performance and address the challenges faced, the following next steps are suggested:

- Conduct feature engineering to identify more informative features that can enhance the model's predictive power.
- Explore ensemble methods to combine the strengths of multiple models and improve overall performance.
- Gather more data, if possible, to increase the representation of the positive class and reduce class imbalance.
- Explore additional techniques to handle class imbalance, such as oversampling or undersampling.
- Optimize the models to improve the recall for the positive class.
- Explore additional evaluation metrics to determine the best model for the classification task.
- Deploy the model to a production environment to make predictions on new data.












## Dependencies
- Python 3.7 or above
- Jupyter Notebook
- Pandas
- Numpy
- Matplotlib
- Seaborn
- Scikit-learn
- Imblearn
- XGBoost


## Usage
1. Clone the repository
2. Install the dependencies
3. Use github [codespaces](https://github.com/HersiYussuf/Capstone_Project)
4. Using the code green button used getting http links for cloning.Create a codespace. 
5. Open the codespace and run the notebook in your favourite editor.


## Contributing


We welcome contributions from the community to improve and enhance our project. If you are interested in contributing, please follow the guidelines below:

1. **Fork** the [repository](https://github.com/HersiYussuf/Capstone_Project.git) by clicking on the "Fork" button.
2. Clone the forked repository to your local machine using the following command:
3. Create a new branch for your feature or bug fix:

### **or**

4. Make your desired changes to the codebase, ensuring they adhere to the project's coding guidelines and best practices.
5. Write appropriate **tests** to cover your changes and ensure that the existing test suite passes successfully.
6. Commit your changes with a clear and descriptive commit message:
7. Push your branch to your forked repository:
8. Submit a **Pull Request (PR)** to the main repository by visiting the [Pull Requests](https://github.com/HersiYussuf/Capstone_Project/pulls) page of the original repository. Outline the changes you have made and provide any relevant information.
9. Engage in discussions with the project maintainers and address any feedback or changes requested.
10. Once approved, your changes will be merged into the main repository.
11. Congratulations! You have successfully contributed to the project.

Please note that by contributing to this project, you agree to abide by the project's [Code of Conduct](https://docs.github.com/en/site-policy/github-terms/github-event-code-of-conduct). Be respectful and considerate when interacting with the community.

If you have any questions or need further assistance, please reach out to us through [contact information or preferred communication channels].

We appreciate your contributions and look forward to your involvement in making this project better!







## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). By contributing to this project, you agree to the terms and conditions of this license.

For more information, please refer to the [LICENSE](https://github.com/HersiYussuf/Capstone_Project/blob/main/LICENSE) file.
## Contributors

- [Aaron Onserio](https://github.com/AaronOnserio)
- [Daniel Ekale](https://github.com/D-EKALE)
- [Emily Njue](https://github.com/EmillyN22)
- [Robert Mbau](https://github.com/robertmbau)
- [Yussuf Hersi](https://github.com/HersiYussuf)
- [Jimcollins Wamae](https://github.com/)
- [Edna Wanjiku](https://github.com/Edna722)



