Calories Burned Prediction Project

Welcome to the Calories Burned Prediction project repository! This project aims to develop a robust machine learning model to accurately predict the number of calories burned during various physical activities based on sensor data. Below is a comprehensive overview of the project, including its objectives, methodology, models used, and results.


Table of Contents

[Project Overview](https://github.com/NamanSingh69/Calorie-Prediction-Model/wiki/Home/_edit#project-overview)

[Dataset](https://github.com/NamanSingh69/Calorie-Prediction-Model/wiki/Home/_edit#dataset)

[Methodology](https://github.com/NamanSingh69/Calorie-Prediction-Model/wiki/Home/_edit#methodology)

[Feature Engineering](https://github.com/NamanSingh69/Calorie-Prediction-Model/wiki/Home/_edit#feature-engineering)

[Modeling](https://github.com/NamanSingh69/Calorie-Prediction-Model/wiki/Home/_edit#modeling)

[Linear Regression](https://github.com/NamanSingh69/Calorie-Prediction-Model/wiki/Home/_edit#linear-regression)

[Random Forest Regression](https://github.com/NamanSingh69/Calorie-Prediction-Model/wiki/Home/_edit#random-forest-regression)

[XGBoost Regression](https://github.com/NamanSingh69/Calorie-Prediction-Model/wiki/Home/_edit#xgboost-regression)

[Neural Network](https://github.com/NamanSingh69/Calorie-Prediction-Model/wiki/Home/_edit#neural-network)

[Model Evaluation](https://github.com/NamanSingh69/Calorie-Prediction-Model/wiki/Home/_edit#model-evaluation)

[Model Interpretability](https://github.com/NamanSingh69/Calorie-Prediction-Model/wiki/Home/_edit#model-interpretability)

[SHAP](https://github.com/NamanSingh69/Calorie-Prediction-Model/wiki/Home/_edit#shap)

[LIME](https://github.com/NamanSingh69/Calorie-Prediction-Model/wiki/Home/_edit#lime)

[Results](https://github.com/NamanSingh69/Calorie-Prediction-Model/wiki/Home/_edit#results)

[Conclusion](https://github.com/NamanSingh69/Calorie-Prediction-Model/wiki/Home/_edit#conclusion)

[References](https://github.com/NamanSingh69/Calorie-Prediction-Model/wiki/Home/_edit#references)


Project Overview

The primary objective of this project is to create an accurate and interpretable predictive model that estimates the number of calories burned during physical activities. This model can be instrumental for fitness enthusiasts, health monitoring systems, and personalized training programs. The project leverages machine learning and deep learning techniques, ensuring the model's robustness and generalizability across diverse user profiles.


Dataset

The project utilizes two main datasets:

Calories.csv: Contains data related to users' calorie expenditure.

Exercise.csv: Includes detailed exercise and physiological data from users.

Both datasets have been merged and preprocessed to ensure data quality and consistency. The final merged dataset comprises 15,000 entries with 8 features relevant to calorie prediction.


Key Features

Gender: Male or Female

Age: Age of the user

Height: Height in centimeters

Weight: Weight in kilograms

Duration: Duration of the activity in minutes

Heart_Rate: Average heart rate during the activity

Body_Temp: Body temperature during the activity

Calories: Calories burned (target variable)


Methodology

The project follows a structured approach encompassing data loading, exploration, preprocessing, feature engineering, model training, evaluation, and interpretability. The steps are as follows:

Data Loading and Merging: Combining the two datasets on the User_ID to create a unified dataset for analysis.

Data Exploration: Analyzing data distributions, checking for missing values, and understanding data types.

Feature Engineering: Creating new features like Body Mass Index (BMI), age groups, and interaction terms to enhance model performance.

Data Preprocessing: Encoding categorical variables, scaling numerical features, and splitting the data into training and testing sets.

Modeling: Implementing various machine learning and deep learning models, including Linear Regression, Random Forest, XGBoost, and Neural Networks.

Model Evaluation: Assessing model performance using metrics like Mean Absolute Error (MAE) and R² Score.

Model Interpretability: Utilizing SHAP and LIME to explain model predictions and understand feature contributions.

Hyperparameter Tuning: Optimizing model parameters to enhance performance.


Feature Engineering

Effective feature engineering is pivotal for enhancing model accuracy. The following features were engineered:

BMI (Body Mass Index): Calculated as weight divided by the square of height (kg/m²).

Age Groups: Categorized users into 'Young', 'Middle-aged', and 'Senior' based on age brackets.

Duration_HeartRate: An interaction term combining duration and heart rate to capture their combined effect on calorie burn.


Modeling

Linear Regression:
A baseline model to establish a reference point for model performance.
Random Forest Regression:
An ensemble learning method using multiple decision trees to improve prediction accuracy and control overfitting.
XGBoost Regression:
An advanced gradient boosting algorithm known for its efficiency and performance in handling large datasets.
Neural Network:
A deep learning model designed to capture complex nonlinear relationships in the data.



Model Evaluation

Models were evaluated using the following metrics:

Mean Absolute Error (MAE): Measures the average magnitude of errors in predictions.

R² Score: Indicates the proportion of variance in the target variable explained by the model.


Insights

XGBoost Tuned Model outperforms all other models with the lowest MAE and highest R² score.

Random Forest models also demonstrate strong performance but are slightly less accurate than XGBoost.

Neural Networks provide good predictions but do not surpass the tree-based models in this setup.


Model Interpretability

Understanding model predictions is crucial for trust and usability. This project employs two powerful interpretability tools:

SHAP (SHapley Additive exPlanations)

Purpose: Provides both global and local explanations of model predictions.

Application: Applied to XGBoost and Random Forest models to visualize feature importance and interaction effects.

Features Identified:

Duration_HeartRate: Most significant predictor.

Duration: Strong positive impact on calorie burn.

Heart_Rate: Also a key contributor.

LIME (Local Interpretable Model-Agnostic Explanations)

Purpose: Offers local explanations by approximating the model locally with an interpretable model.

Application: Applied to the Neural Network model to explain individual predictions.

Key Takeaways

SHAP effectively highlights the most influential features across models, enhancing understanding of model behavior.

LIME provides granular insights into individual predictions, albeit with initial preprocessing challenges that were subsequently addressed.


Results
The project achieved remarkable results, particularly with the XGBoost Tuned Model, which demonstrated exceptional accuracy in predicting calories burned with an MAE of 1.14 and an R² score of 0.99. This indicates near-perfect predictions, showcasing the model's capability to generalize and perform effectively on unseen data.

Neural Network Performance:
MAE of 1.47 and R² of 1.00.


Conclusion

This project successfully developed a highly accurate model for predicting calories burned using advanced machine learning techniques. The XGBoost Tuned Model stood out as the most effective, providing reliable and precise predictions. Model interpretability tools like SHAP and LIME were instrumental in understanding feature impacts, fostering trust in the model's decisions.


Key Achievements

Accurate Predictions: Achieved low MAE and high R² scores with tree-based models.
Comprehensive Feature Engineering: Enhanced model performance through thoughtful feature creation and selection.
Model Interpretability: Utilized SHAP and LIME to demystify model predictions, ensuring transparency.
Hyperparameter Optimization: Fine-tuned models to extract optimal performance.


References
Scikit-learn Documentation: https://scikit-learn.org/stable/documentation.html
XGBoost Documentation: https://xgboost.readthedocs.io/en/latest/
TensorFlow Keras Documentation: https://www.tensorflow.org/guide/keras
SHAP Documentation: https://shap.readthedocs.io/en/latest/
LIME Documentation: https://lime-ml.readthedocs.io/en/latest/
