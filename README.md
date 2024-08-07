# Obesity-Classification-with-LightGBM
Multi-class classification task of determining an individual's level (or lack) of obesity, using LightGBM

![image](https://github.com/user-attachments/assets/b9f42ee6-1ff5-4637-a9ec-9b2ab0feb5cf)

---

### Objective
This project was completed as part of CU Boulder's Data Mining Project Course and as part of Kaggle's Playground Series (Season 4, Episode 2). This project aims to identify significant predictors of obesity and will do so by leveraging a LightGBM classifier model to predict the level of obesity (or lack of obesity) in an individual given their predictive characteristics. LightGBM is a fast, distributed, high performance gradient boosting framework based on decision tree alorithms and can be used for classification tasks. The project utilizes two datasets, one from Kaggle (synthetically generated) and one from ARAVINDPCODER (the original dataset). Both datasets include the categorical target variable 'NObeyesdad' and 16 predictor variables (some categorical and others numerical). The target variable has the following levels: Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III. 

---

### Methods
Libraries Used
- pandas
- numpy
- seaborn
- matplotlib
- sklearn (train_test_split, accuracy_score, confusion_matrix, classification_report)
- optuna (TPESampler)
- lightgbm (LGBMClassifier, plot_importance)

Exploratory Data Analysis (EDA)
- Summary statistics of both the training and original datasets
- Donut charts visualizing the distributions of the 8 categorical predictor variables and the target variable
- Kernet density estimate plots visualizing the distributions of the 8 numerical predictor variables
- Heatmap showing potential correlations between numerical predictors

Data Preprocessing
- Extracting variable types (continuous/numerical & categorical)
- Removing 'id' column from both train and test datasets
- Concatenating the train and original datasets
- One-hot encoding categorical variables in both train and test datasets
- Separating the predictors (X) and target variable (Y)
- Splitting the data into training (80%) and testing (20%) sets

Building the Model
- Leveraging Optuna (open source hyperparameter optimization framework) to determine the optimal parameter values for the LightGBM model
- 50 trials to determine the optimal parameter values for: learning_rate, n_estimators, lambda_l1 & lambda_l2, max_depth, colsample_bytree, subsample, min_child_samples
- objective = multiclass
- metric = multi_logloss
- boosting_type = gbdt
- random_state = 42
- num_class = 7

Model Evaluation
- Checking the accuracy score of the final model
- Checking for overfitting
- Viewing a classification report (precision, recall, f1-score, support)
- Viewing a confusion matrix
- Feature importance

---

### General Results
Final Model Accuracy
- 91.7%

Feature Importance
- Weight, Height, Age

The final LightGBM model yielded an accuracy score of ~91.7%. The model did not appear to overfit the data, as the training set accuracy score (0.9501) did not differ greatly from the testing set accuracy score (0.9173). The classification report and confusion matrix suggested that the model performed best when predicting the 'Obesity_Type_III' class (875 correct predictions, 3 incorrect predictions). The model appeared to perform the worst when predicting the 'Overweight_Level_I' class (430 correct, 95 incorrect). Lastly, a feature importance plot showed that the three most important predictors were 'Weight', 'Height', and 'Age'. 
