# Proposed Research Novelty and Project Extension

## 1. Project Overview

The project focuses on predicting hospital readmission among diabetic patients using machine learning techniques. It integrates prediction functionality into a web application using a Flask backend and an Angular frontend. GitHub Actions supports automated testing and build workflows, while GitHub Pages is used to deploy the frontend.

The research extends the original study by improving patient-level validation, evaluating model performance across different age groups, and proposing severity-based readmission risk categories.

## 2. Identified Research Gap

1. **Lack of Patient-Level Separation:** The original paper used cross-validation without enforcing patient-level separation. Records belonging to the same patient could therefore appear in both training and validation sets, potentially leading to overly optimistic performance estimates.

2. **Lack of Age-Group-Based Evaluation:** The original paper did not evaluate model performance separately across different age groups, making it difficult to understand how well the model performs for different age categories.

3. **Lack of Severity-Based Risk Classification:** The project proposes categorizing predicted readmission risk into mild, moderate, and severe levels to make prediction results easier to interpret.

## 3. Proposed Solution

1. **Patient-Level Cross-Validation:** GroupKFold cross-validation keeps records belonging to the same patient in the same fold, preventing those records from appearing in both the training and validation sets.

2. **Age-Group Performance Analysis:** Model performance is evaluated separately across different age groups to identify differences in prediction performance.

3. **Severity-Based Risk Classification:** Predicted readmission probabilities can be mapped to mild, moderate, and severe risk categories using defined thresholds. These categories must be validated and should represent readmission risk rather than medical severity.

## 4. Project Novelty

The project aims to improve the reliability and interpretability of diabetes readmission prediction through patient-level cross-validation and age-group-based performance analysis. It also proposes severity-based risk categories to make predicted readmission risk easier to understand.

These contributions focus on addressing evaluation limitations and improving the presentation of model predictions.

## 5. Expected Contribution

- Reduce the risk of data leakage during model evaluation.
- Provide a clearer understanding of model performance across different age groups.
- Improve the interpretability of predictions through risk categories, once implemented and validated.
- Support a more reliable evaluation of machine learning models for diabetes readmission prediction.
