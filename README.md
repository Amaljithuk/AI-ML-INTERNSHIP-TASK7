Breast Cancer Classification with Support Vector Machines (SVM)
Task Overview
This repository contains the solution for Task 7 of the Elevate AI & ML Internship. The objective is to implement SVM classifiers with linear and RBF kernels on the Breast Cancer Dataset to predict diagnosis (Malignant or Benign).
Steps Performed

Data Preprocessing:
Loaded breast-cancer.csv, dropped id column, and normalized numerical features.
Encoded diagnosis as numerical labels (0: Benign, 1: Malignant).
Saved cleaned dataset as Breast_Cancer_Cleaned.csv.


Exploratory Data Analysis:
Generated summary statistics, class distribution plot, and correlation heatmap.
Created a pairplot for top features (radius_mean, texture_mean, perimeter_mean, area_mean).


SVM Classifier:
Trained SVM models with linear and RBF kernels, tuning hyperparameters (C, gamma) using GridSearchCV.
Evaluated using 5-fold cross-validation, accuracy, and confusion matrix.


Decision Boundary Visualization:
Visualized decision boundaries using radius_mean and texture_mean.



Files

breast_cancer_svm_classification.ipynb: Jupyter Notebook with preprocessing, EDA, and SVM implementation.
breast-cancer.csv: Original Breast Cancer Dataset.
Breast_Cancer_Cleaned.csv: Preprocessed dataset.
class_distribution.png: Bar plot of class distribution.
correlation_heatmap.png: Heatmap of feature correlations.
pairplot.png: Pairplot of top features by diagnosis.
confusion_matrix_linear.png: Confusion matrix for linear kernel.
confusion_matrix_rbf.png: Confusion matrix for RBF kernel.
decision_boundary_linear.png: Decision boundary for linear kernel.
decision_boundary_rbf.png: Decision boundary for RBF kernel.
interview_questions.md: Answers to provided interview questions.
README.md: This documentation file.

Tools Used

Python 3
Jupyter Notebook
Pandas
NumPy
Matplotlib
Seaborn
scikit-learn

How to Run

Clone this repository:git clone (https://github.com/Amaljithuk/AI-ML-INTERNSHIP-TASK7)


Install required dependencies:pip install pandas numpy matplotlib seaborn scikit-learn jupyter


Launch Jupyter Notebook:jupyter notebook


Open breast_cancer_svm_classification.ipynb and run all cells to:
Generate the cleaned dataset (Breast_Cancer_Cleaned.csv).
Save visualizations (class_distribution.png, correlation_heatmap.png, pairplot.png, confusion_matrix_linear.png, confusion_matrix_rbf.png, decision_boundary_linear.png, decision_boundary_rbf.png).
Display model metrics.



Observations

EDA: The dataset is slightly imbalanced (more Benign cases). Features like radius_mean and texture_mean show discriminative patterns.
SVM Performance: Both linear and RBF kernels achieve high accuracy, with RBF often outperforming due to non-linear patterns. Cross-validation ensures robust generalization.
Decision Boundaries: Linear kernel shows a straight boundary, while RBF captures more complex patterns in the 2D feature space.
Normalization: Critical for SVM to ensure fair feature contributions in distance calculations.

Submission
This repository is submitted for Task 7 of the Elevate AI & ML Internship, completed on July 4, 2025, within the 10:00 AM to 10:00 PM submission window.
