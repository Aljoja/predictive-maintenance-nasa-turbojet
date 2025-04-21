# Predictive Maintenance - Remaining Useful Life (RUL) Prediction

## **Table of Contents**
- [Predictive Maintenance - Remaining Useful Life (RUL) Prediction](#predictive-maintenance---remaining-useful-life-rul-prediction)
  - [**Table of Contents**](#table-of-contents)
  - [**Project Overview**](#project-overview)
  - [**Purpose**](#purpose)
  - [**Dataset Information**](#dataset-information)
    - [**Dataset Files**:](#dataset-files)
    - [**Data Columns**:](#data-columns)
    - [**Why the Data is Split into Train and Test**:](#why-the-data-is-split-into-train-and-test)
    - [**What the Data Represents**:](#what-the-data-represents)
  - [**Project Structure**](#project-structure)
  - [**Models and Techniques**](#models-and-techniques)
    - [**1. Random Forest**](#1-random-forest)
    - [**2. XGBoost**](#2-xgboost)
    - [**3. LSTM (Long Short-Term Memory)**](#3-lstm-long-short-term-memory)
    - [**4. PyTorch (in progress)**](#4-pytorch-in-progress)
    - [**Other Models Considered**:](#other-models-considered)
  - [**Training and Evaluation Techniques**](#training-and-evaluation-techniques)
  - [**How to Run the Project**](#how-to-run-the-project)
  - [**Results and Analysis**](#results-and-analysis)
  - [**Future Improvements**](#future-improvements)
  - [**License**](#license)

## **Project Overview**

This project aims to predict the **Remaining Useful Life (RUL)** of aircraft engines based on sensor data. The goal is to create a predictive maintenance system capable of forecasting the number of cycles remaining before an engine fails. This can help in planning maintenance schedules, reducing downtime, and improving the overall safety and efficiency of operations.

## **Purpose**

The purpose of this project is to:
1. Predict the **Remaining Useful Life (RUL)** of engines using sensor data.
2. Train and evaluate different machine learning models to determine the most effective model for predicting RUL.
3. Provide insights into predictive maintenance techniques that can be applied to other industrial applications.

## **Dataset Information**

The dataset used in this project comes from the **C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset, which contains time-series data from a set of simulated turbofan engines. The data is provided by NASA as part of a challenge to predict the RUL of aircraft engines.

### **Dataset Files**:
The data is split into **training** and **test** sets for multiple engine configurations:
- **`train_FD001.txt`**: Data from 100 engines under **sea level conditions**, with **HPC degradation** as the fault mode.
- **`train_FD002.txt`**: Data from 260 engines under **six operational conditions**, with **HPC degradation** as the fault mode.
- **`train_FD003.txt`**: Data from 100 engines under **sea level conditions**, with **two fault modes**: **HPC degradation** and **Fan degradation**.
- **`train_FD004.txt`**: Data from 248 engines under **six operational conditions**, with **two fault modes**: **HPC degradation** and **Fan degradation**.

Each dataset contains sensor readings and operational settings for each engine, along with the **Remaining Useful Life (RUL)** for the training data.

### **Data Columns**:
- **`unit_number`**: The engine unit ID.
- **`time_in_cycles`**: The number of cycles the engine has been in operation.
- **`operational_setting_1, operational_setting_2, operational_setting_3`**: Parameters that influence engine performance.
- **`sensor_1, sensor_2, ..., sensor_26`**: Measurements from various sensors installed on the engine (e.g., pressure, temperature, etc.).
- **`RUL`**: Remaining Useful Life (RUL) of the engine (only in training datasets).

### **Why the Data is Split into Train and Test**:
- **Training Data**: The model uses the training dataset (which contains sensor readings and RUL) to learn the relationship between input features (sensor data) and the target variable (RUL).
- **Test Data**: The test dataset is used to evaluate how well the trained model generalizes to unseen data. It contains only sensor readings (no RUL), and the model's task is to predict the RUL for these engines.

### **What the Data Represents**:
- **Train Data**: Data from engines operating under different conditions with known RUL values, used for training the models.
- **Test Data**: Data from engines with unknown RUL values, used for predicting the RUL and evaluating model performance.

## **Project Structure**

```plaintext
project/
├── data/
│   ├── raw/               # Original, unmodified raw data files
│   └── processed/         # Cleaned and processed data ready for training
├── notebooks/             # Jupyter notebooks for exploration and testing
│   ├── 01_data_exploration.ipynb  # Data exploration and preprocessing
│   ├── 02_feature_engineering.ipynb # Feature engineering
│   └── 03_model_training.ipynb  # Model training and evaluation
├── src/                   # Source code for models, preprocessing, and training
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_definition.py
│   ├── train.py
│   └── inference.py
├── models/                # Folder to store trained models
├── requirements.txt       # Python dependencies
├── environment.yml        # Conda environment configuration
└── README.md              # Project documentation
```

## **Models and Techniques**

This project uses various machine learning and deep learning models to predict the **RUL** of the engines. The models and techniques used are as follows:

### **1. Random Forest**
- A powerful ensemble model based on decision trees, used to capture complex non-linear relationships in the data.

### **2. XGBoost**
- A popular gradient boosting framework known for its high performance and efficiency. It works well for structured data and is capable of handling missing values and different feature types.

### **3. LSTM (Long Short-Term Memory)**
- A type of Recurrent Neural Network (RNN) that is well-suited for time-series data. LSTM models can capture dependencies over time, making them ideal for predicting RUL from sequential sensor readings.

### **4. PyTorch (in progress)**
- **PyTorch** will be used alongside **TensorFlow** to explore deep learning approaches. The models will focus on time-series forecasting using **LSTM** and potentially **1D CNN-LSTM hybrid models** for improved performance.

### **Other Models Considered**:
- **Gradient Boosting Machines (GBM)**
- **Support Vector Machines (SVM)**
- **Random Forest Regressor**
- **Neural Networks (ANNs)** for non-time-series models

## **Training and Evaluation Techniques**

1. **Hyperparameter Tuning**: 
   We use **GridSearchCV** for Random Forest and XGBoost to optimize the hyperparameters for better model performance.
   For the **LSTM** model, manual tuning of the **epochs** and **batch sizes** is used.

2. **Cross-Validation**:
   Cross-validation is used to evaluate model performance more robustly. This helps in understanding the model's generalization ability across different subsets of the data.

3. **Evaluation Metrics**:
   The models are evaluated using two primary metrics:
   - **MAE (Mean Absolute Error)**: Measures the average magnitude of errors between predicted RUL and actual RUL.
   - **RMSE (Root Mean Squared Error)**: Measures the standard deviation of the prediction errors, providing insight into how spread out the predictions are.

## **How to Run the Project**

1. **Install Dependencies**:
   - Use the `requirements.txt` file or `environment.yml` to set up the environment.
   - Example for pip:
     ```bash
     pip install -r requirements.txt
     ```

   - Example for conda:
     ```bash
     conda env create -f environment.yml
     ```

2. **Run Jupyter Notebooks**:
   - Open and run the Jupyter notebooks in the **`notebooks/`** folder for data exploration, feature engineering, and model training.

3. **Training the Models**:
   - Run **`train.py`** to train and tune the models.
     ```bash
     python src/train.py --train-data data/raw/train_FD001.txt --test-data data/raw/test_FD001.txt --model random_forest
     ```

4. **Running Inference**:
   - Use **`inference.py`** to load the trained model and make predictions.
     ```bash
     python src/inference.py --model models/random_forest_model.pkl --test-data data/raw/test_FD001.txt
     ```

## **Results and Analysis**

The models will be evaluated based on **MAE** and **RMSE**. The **Random Forest** and **XGBoost** models showed promising results, while the **LSTM** model's performance will be refined further with more hyperparameter tuning and the use of **PyTorch** for comparison.

## **Future Improvements**

1. Experiment with hybrid models (e.g., **1D CNN + LSTM**) for better feature extraction from time-series data.
2. Implement **deep reinforcement learning** to predict RUL more effectively.
3. Apply **Transfer Learning** to leverage knowledge from similar datasets or fault modes.

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
