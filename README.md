# Calories Burnt Prediction Model

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

## Table of Contents

- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Explanation](#model-explanation)
- [Model Interpretability](#model-interpretability)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation Metrics](#evaluation-metrics)
- [Saving and Deployment](#saving-and-deployment)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

The **Calories Burnt Prediction Model** is a comprehensive machine learning project aimed at predicting the number of calories burned by individuals based on various personal and exercise-related metrics. Utilizing advanced machine learning techniques and deep learning models, the project demonstrates the complete pipeline from data acquisition and preprocessing to model training, evaluation, and deployment. Additionally, the project emphasizes model interpretability to ensure transparency and trust in the predictions.

## Data Description

The project utilizes two primary datasets sourced from Kaggle:

1. **`exercise.csv`**: Contains detailed exercise metrics for users, including demographic and physiological information.
2. **`calories.csv`**: Records the number of calories burned by users during their exercise sessions.

Both datasets are merged on the `User_ID` field to create a unified dataset for analysis and modeling.

### Dataset Fields

- **`exercise.csv`**
  - `User_ID`: Unique identifier for each user.
  - `Gender`: Gender of the user (`male` or `female`).
  - `Age`: Age of the user in years.
  - `Height`: Height of the user in centimeters.
  - `Weight`: Weight of the user in kilograms.
  - `Duration`: Duration of the exercise session in minutes.
  - `Heart_Rate`: Average heart rate during the exercise.
  - `Body_Temp`: Body temperature of the user during exercise.

- **`calories.csv`**
  - `User_ID`: Unique identifier for each user.
  - `Calories`: Number of calories burned during the exercise session.

## Project Structure

```
Calories-Burnt-Prediction/
├── Calories Burnt Prediction Model.pdf
├── calories.csv
├── exercise.csv
├── mlml.ipynb
├── calories_burnt_xgb_model.pkl
├── scaler.pkl
├── README.md
└── requirements.txt
```

### File Descriptions

- **`Calories Burnt Prediction Model.pdf`**
  - A comprehensive PDF document containing the entire project code along with outputs, visualizations, and detailed explanations.

- **`calories.csv`**
  - Dataset containing calorie burn records for users.

- **`exercise.csv`**
  - Dataset containing exercise-related metrics for users.

- **`mlml.ipynb`**
  - The main Jupyter Notebook (`mlml.ipynb`) that hosts the project's code, including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, interpretability, and deployment steps.

- **`calories_burnt_xgb_model.pkl`**
  - The serialized (pickled) trained XGBoost regression model, optimized for predicting calories burned.

- **`scaler.pkl`**
  - The serialized `StandardScaler` object used for scaling numerical features, ensuring consistent preprocessing during inference.

- **`README.md`**
  - This README file providing an overview and instructions for the project.

- **`requirements.txt`**
  - A list of all Python dependencies required to run the project. *(Note: If not present, you may need to create one based on the imported libraries in `mlml.ipynb`.)*

## Installation

To set up the project locally, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/NamanSingh69/Calories-Burnt-Prediction.git
   cd Calories-Burnt-Prediction
   ```

2. **Create a Virtual Environment:**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   Install the required Python packages using `pip`.

   ```bash
   pip install -r requirements.txt
   ```

   *If `requirements.txt` is not provided, you can install the necessary packages manually:*

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow scikeras shap lime joblib
   ```

4. **Download Datasets:**

   Ensure that both `exercise.csv` and `calories.csv` are placed in the project directory. These can be obtained from Kaggle or the provided datasets.

## Usage

### Running the Jupyter Notebook

1. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

2. **Open `mlml.ipynb`:**

   Navigate to the `mlml.ipynb` notebook and execute the cells sequentially to run the project.

### Making Predictions with the Saved Model

The project includes a function to make predictions on new data using the trained XGBoost model.

1. **Ensure Dependencies are Installed:**

   Make sure all required packages are installed as per the [Installation](#installation) section.

2. **Load the Model and Scaler:**

   The `calories_burnt_xgb_model.pkl` and `scaler.pkl` files are essential for making predictions. Ensure they are present in the project directory.

3. **Use the Prediction Function:**

   Here's an example of how to use the prediction function:

   ```python
   import pandas as pd
   import joblib

   def predict_calories(input_data):
       """
       Predicts the number of calories burned given input data.
       
       Parameters:
       input_data (dict): Dictionary containing input data with keys:
           - Gender (str): 'male' or 'female'
           - Age (float)
           - Height (float)
           - Weight (float)
           - Duration (float)
           - Heart_Rate (float)
           - Body_Temp (float)
       
       Returns:
       float: Predicted calories burned.
       """
       # Create DataFrame from input data
       input_df = pd.DataFrame([input_data])

       # Feature Engineering
       input_df['BMI'] = input_df['Weight'] / (input_df['Height']/100)**2
       input_df['Duration_HeartRate'] = input_df['Duration'] * input_df['Heart_Rate']
       input_df['Age_Group'] = pd.cut(input_df['Age'], bins=[0, 30, 50, 100], labels=['Young', 'Middle-aged', 'Senior'])

       # One-Hot Encoding
       input_df = pd.get_dummies(input_df, columns=['Gender', 'Age_Group'], drop_first=True)

       # Ensure all features are present in the same order as training data
       missing_cols = set(['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI', 
                           'Duration_HeartRate', 'Gender_male', 'Age_Group_Middle-aged', 'Age_Group_Senior']) - set(input_df.columns)
       for col in missing_cols:
           input_df[col] = 0
       input_df = input_df[['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 
                            'BMI', 'Duration_HeartRate', 'Gender_male', 'Age_Group_Middle-aged', 
                            'Age_Group_Senior']]

       # Load the scaler
       scaler = joblib.load('scaler.pkl')

       # Scale numerical features
       numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 
                             'BMI', 'Duration_HeartRate']
       input_df[numerical_features] = scaler.transform(input_df[numerical_features])

       # Load the saved model
       model = joblib.load('calories_burnt_xgb_model.pkl')

       # Predict
       predicted_calories = model.predict(input_df)

       return predicted_calories[0]

   # Sample input data
   sample_input = {
       'Gender': 'female',
       'Age': 30,
       'Height': 185,
       'Weight': 75,
       'Duration': 45,
       'Heart_Rate': 120,
       'Body_Temp': 38.5
   }

   # Predict calories
   predicted_calories = predict_calories(sample_input)

   print(f"Predicted Calories Burned: {predicted_calories:.2f}")
   ```

   **Output:**
   ```
   Predicted Calories Burned: 226.56
   ```

## Model Explanation

The project explores various machine learning models to predict calories burned, including:

1. **Linear Regression**
   - **Description:** A baseline regression model that assumes a linear relationship between features and the target variable.
   - **Performance:** Achieved a Mean Absolute Error (MAE) of ~5.72 and an R² score of ~0.98.

2. **Random Forest Regressor**
   - **Description:** An ensemble learning method using multiple decision trees to improve predictive performance.
   - **Performance:** Initially achieved an MAE of ~1.71 and an R² score of ~1.00. After hyperparameter tuning, MAE slightly improved to ~1.69.

3. **XGBoost Regressor**
   - **Description:** An optimized gradient boosting framework known for its efficiency and performance.
   - **Performance:** Initially achieved an MAE of ~1.36 and an R² score of ~1.00. After hyperparameter tuning, MAE further improved to ~1.14.

4. **Neural Network (Deep Learning)**
   - **Description:** A sequential neural network with dense layers and dropout regularization to predict the target variable.
   - **Performance:** Achieved an MAE of ~1.47 and an R² score of ~1.00.

### **Conclusion:**

- **Best Performer:** The tuned XGBoost model outperformed all other models with the lowest MAE (~1.14) and an R² score close to 1.00, indicating excellent predictive capability.
- **Model Selection:** Despite similar high R² scores across models, MAE provides a clearer understanding of average prediction errors, making XGBoost the preferred choice.

## Model Interpretability

Understanding how models make predictions is crucial for trust and transparency. This project leverages SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations) for model interpretability.

### SHAP

- **Global Interpretation:** SHAP summary plots and dependence plots reveal the overall feature importance and their interactions.
- **Local Interpretation:** SHAP force plots explain individual predictions by showing the contribution of each feature.

### LIME

- **Instance-Level Explanation:** Provides explanations for specific predictions, highlighting which features influenced the model's decision the most for a given instance.

## Hyperparameter Tuning

Optimizing model hyperparameters is essential to enhance performance. This project employs `GridSearchCV` for hyperparameter tuning of the Random Forest and XGBoost models.

### Random Forest

- **Parameters Tuned:**
  - `n_estimators`: [100, 200, 300]
  - `max_depth`: [None, 15, 30]
  - `min_samples_split`: [2, 5]
  - `min_samples_leaf`: [1, 2]
  - `bootstrap`: [True, False]
- **Best Parameters:**
  ```python
  {'bootstrap': True, 'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300}
  ```
- **Tuned Performance:**
  - **Test MAE:** ~1.69
  - **R² Score:** ~1.00

### XGBoost

- **Parameters Tuned:**
  - `n_estimators`: [100, 200]
  - `learning_rate`: [0.01, 0.1]
  - `max_depth`: [3, 5]
  - `subsample`: [0.8, 1.0]
  - `colsample_bytree`: [0.8, 1.0]
- **Best Parameters:**
  ```python
  {'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 200, 'subsample': 0.8}
  ```
- **Tuned Performance:**
  - **Test MAE:** ~1.14
  - **R² Score:** ~1.00

## Evaluation Metrics

The following metrics are used to evaluate model performance:

- **Mean Absolute Error (MAE):** Measures the average magnitude of errors in predictions, without considering their direction. Lower MAE indicates better performance.
  
- **R² Score:** Represents the proportion of variance in the target variable explained by the model. An R² score close to 1 indicates a strong fit.

## Saving and Deployment

The project includes mechanisms to save the trained model and scaler for deployment:

- **Model Saving:**
  - **`calories_burnt_xgb_model.pkl`**: Serialized XGBoost model.
  
- **Scaler Saving:**
  - **`scaler.pkl`**: Serialized `StandardScaler` object for consistent feature scaling during inference.

### Making Predictions with New Data

A prediction function `predict_calories` is provided to generate calorie burn predictions for new user data. This function handles data preprocessing, feature engineering, scaling, and model inference seamlessly.

**Example Usage:**

```python
# Sample input data
sample_input = {
    'Gender': 'female',
    'Age': 30,
    'Height': 185,
    'Weight': 75,
    'Duration': 45,
    'Heart_Rate': 120,
    'Body_Temp': 38.5
}

# Predict calories
predicted_calories = predict_calories(sample_input)

print(f"Predicted Calories Burned: {predicted_calories:.2f}")
```

**Output:**
```
Predicted Calories Burned: 226.56
```

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

2. **Create a New Branch**
   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**
   ```bash
   git commit -m "Add your feature"
   ```

4. **Push to the Branch**
   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## Acknowledgements

- **Kaggle:** For providing the datasets.
- **Community Contributors:** For their valuable insights and support.
