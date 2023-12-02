# CS_577

# Pioneering Heart Disease 

This project aims to predict the presence of heart disease based on various medical attributes using machine learning models.

## Overview

Heart disease is one of the leading causes of mortality worldwide. Early prediction and diagnosis play a crucial role in effective treatment and prevention. This project utilizes a dataset containing several medical attributes to develop and compare machine learning models for predicting the likelihood of heart disease.

## Dataset

The dataset used in this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease). It consists of several features including age, sex, cholesterol levels, resting blood pressure, and more, contributing to the prediction of heart disease.

## Models Developed

### 1. Random Forest
- Utilized RandomForestClassifier from scikit-learn with 100 estimators for prediction.

### 2. Naive Bayes
- Implemented Gaussian Naive Bayes classifier for heart disease prediction.

### 3. K-Nearest Neighbors (KNN)
- Utilized the KNeighborsClassifier algorithm with k=5 for classification.

### 4. Multi-layer Perceptron (MLP)
- Developed an Artificial Neural Network (ANN) using MLPClassifier with hidden layers of size 100 and 50.

### 5. Convolutional Neural Network (CNN)
- Explored a CNN architecture, although its suitability for this dataset is limited due to the non-image nature of the data.

### 6. Logistic Regression
- Trained a logistic regression model for comparison with other machine learning models.

### 7. Recurrent Neural Network (RNN)
- Implemented a simple RNN model, although RNNs might not be the optimal choice for tabular data like heart disease prediction.

### 8. Artificial Neural Network (ANN) 
- Developed an Artificial Neural Network (ANN) using the Classifier with hidden layers of size 100 and 50, achieving an accuracy of 85% on the heart disease dataset.

## Analysis

- Feature importance analysis conducted using the Random Forest model indicated that OLDPEAK was the primary determinant for heart disease prediction.

## Usage

- To use this project, clone the repository and run the provided scripts in Python. Ensure the necessary libraries are installed by using the `requirements.txt` file.
- Data preprocessing, model development, and evaluation scripts are included in the repository.

## Results

- The accuracy scores of the developed models are as follows:
  - Random Forest: 75.93%
  - Naive Bayes: 90.74%
  - K-Nearest Neighbors: 81.48%
  - MLP: 85.19%
  - Logistic Regression: 90.74%
  - CNN: 61.11%
  - RNN: 41.0%
  - ANN: 85.19


## Conclusion

This project aimed to predict heart disease using various deep learning models. Through rigorous analysis and comparison, it was found that the there are five models exhibited the highest accuracy in predicting heart disease from the provided dataset. However, the suitability of models varied, with simpler algorithms like Logistic Regression also providing competitive results. The analysis highlights the importance of early prediction using deep learning combined with the ML in identifying potential cases of heart disease. Further improvements and explorations can be made in feature engineering and model tuning to enhance predictive performance.

## Acknowledgments

- Credit to the UCI Machine Learning Repository for providing the heart disease dataset.
- Credit to our Professor who has provided this oppurtunity for working on this project.
- Credit to all the T.A's for supporting us from the beginning of the work.

Feel free to contribute, raise issues, or suggest improvements!
