
# Diabetes Prediction

## Problem Setting

Diabetes is a chronic (long-lasting) health condition that affects how your body turns food into
energy. Your body breaks down most of the food you eat into sugar (glucose) and releases it into
your bloodstream. When your blood sugar goes up, it signals your pancreas to release insulin.
Insulin acts like a key to let the blood sugar into your body’s cells for use as energy. With
diabetes, your body doesn’t make enough insulin or can’t use it as well as it should. When there
isn’t enough insulin or cells stop responding to insulin, too much blood sugar stays in your
bloodstream. Over time, that can cause serious health problems, such as heart disease, vision
loss, and kidney disease. The World Health Organization (WHO) estimates that 422 million
people worldwide have diabetes, mostly in low- or middle-income nations. And up until the year
2030, this might be increased to 490 billion.

## Problem definition

The only method of preventing diabetes complications is to identify and treat the disease early.
The early detection of diabetes is important because its complications increase over time. Also,
prediction of diabetes at an early stage can lead to improved treatment. The intention of this
project is to build supervised models like Logistic regression, K nearest neighbors, Support
Vector Machines, Random Forest and Decision trees and select the best algorithm which can
perform early prediction of diabetes for a patient with high accuracy and analyze the variables
which are more responsible for causing diabetes.

## Data Source

The dataset for diabetes prediction has been taken from Centers for Disease Control and
Prevention (CDC) submitted by National Center for Health statistics (NCHS) as part of National
Health and Nutrition Examination Survey (NHANES) conducted in the year 2013-2014.
(https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2013)

## Data Description

This dataset comprises a total of 6643 instances joined from multiple datasets such as Diet data,
Lab data, Demographic data. There are a total of 21 attributes some of which are Age, BMI,
Systolic Blood Pressure, Total Cholesterol, Glycohemoglobin and 1 target attribute – ‘Diabetes’.
A more detailed description of all the variables is given in the below table.

![Image](https://github.com/anudeep-ga/Diabetes_Prediction/blob/main/Images/data-description.png?raw=true)

## Model Selection

### 1. Logistic Regression:
Parameters used: Cs = 100, penalty = 'l2', random_state = 7

The logistic regression model was trained using cross-validation and grid search to optimize
the hyperparameters. The model's performance was evaluated using sensitivity, specificity,
accuracy, AUC score, and F1-score. The results indicate that the model performed well, with
a sensitivity of 0.989011 and specificity of 0.993374. The model accurately identified
positive and negative cases with an accuracy of 0.992975, and the AUC score of 0.999117
indicates that the model can differentiate between positive and negative cases effectively.
The F1-score of 0.962567 represents the model's overall performance, which is good.
The grid search was performed with a five-fold cross-validation strategy, and the best value
for the regularization parameter (c) was found to be 100, resulting in the highest accuracy of
the model. The use of StratifiedKFold with three splits ensured that the distribution of the
target variable was maintained in each fold.
In conclusion, the logistic regression model trained with cross-validation and grid search
performed well, and the best value for the hyperparameter (c) was found to be 100. The use
of cross-validation and grid search helped to optimize the model's hyperparameters and
improve its performance, making it an effective tool for identifying positive and negative
cases.

Hyper Parameter Tuning:

The logistic regression model was optimized using cross-validation and grid search to find
the best hyperparameters. The regularization parameter (C) was tuned, and the best value was
found to be 100. A five-fold cross-validation strategy was used during grid search, and
StratifiedKFold with three splits was used to maintain target variable distribution. The
optimized model showed improved performance, with high sensitivity, specificity, accuracy,
AUC score, and F1-score.


### 2. Naïve Bayes Classifier:
The model's sensitivity and specificity values were 0.741758 and 0.899503, respectively,
indicating that the model was able to accurately identify both positive and negative instances.
However, these values were lower than the corresponding values for the Decision Tree
model. The overall accuracy of the Naive Bayes model was 88.5%, which was lower than the
accuracy of the KNN model. The model's AUC score was 0.877486, indicating that the
model was able to distinguish between positive and negative instances reasonably well. The
F1-score of the model was found to be 0.541082, indicating that the model had lower balance
between precision and recall than the KNN model. Overall, the Naive Bayes model exhibited
lower performance than the Decision Tree model in predicting diabetes using the given
dataset. However, the model still exhibited moderate accuracy and reasonable ability to
distinguish between positive and negative instances.

### 3. k-NN Classifier:

Parameters used: n
_
neighbors = 2

The best value for the hyperparameter k in KNN model was found to be 2 through cross-
validation. This suggests that the model may be overfitting when k is set to higher values,
and a value of 2 provides the best balance between bias and variance in the model. The
sensitivity of the model is 0.736, which means that the model correctly identifies 73.6% of
the positive class instances. The specificity of the model is 0.975, which means that the
model correctly identifies 97.5% of the negative class instances. The accuracy of the model is
0.953, which means that the model correctly classifies 95.3% of all instances in the dataset.
The AUC score of the model is 0.901, which is a measure of the model's ability to distinguish
between positive and negative classes. The score ranges from 0 to 1, with a higher score
indicating better performance. The AUC score of 0.901 suggests that the model is performing
well in distinguishing between positive and negative instances. The F1-score of the model is
0.742, which is a measure of the model's balance between precision and recall. A high F1-
score indicates a model with high precision and recall.

### 4. Decision Tree:
Parameters used: max_depth=21, criterion='entropy', random_state=100

The hyperparameters max depth and impurity method were found to be optimized at 15 and
entropy respectively, through cross-validation. This suggests that a deeper tree with the
entropy impurity method provides the best balance between bias and variance in the model.
The sensitivity of the model is 0.818, which means that the model correctly identifies 81.8%
of the positive class instances. The specificity of the model is 0.956, which means that the
model correctly identifies 95.6% of the negative class instances. The accuracy of the model is
0.943, which means that the model correctly classifies 94.3% of all instances in the dataset.
The AUC score of the model is 0.887, which is a measure of the model's ability to distinguish
between positive and negative classes. The F1-score of the model is 0.725, which is an
optimized measure of the model's balance between precision and recall.

### 5. Random Forest:
The Random Forest model's performance was evaluated for a classification task using
metrics such as sensitivity, specificity, accuracy, AUC score, and F1-score. The model
showed moderate performance, with a sensitivity value of 0.824176 indicating the correct
identification of true positive cases, and specificity value of 0.96963 indicating the accurate
identification of true negative cases. The model's overall accuracy was 0.956347, indicating
the correct identification of the majority of cases. AUC score of 0.98284 suggested the
model's ability to differentiate between positive and negative cases, while F1-score of
0.775194 indicated the model's average overall performance. The Random Forest model is
known for its ability to handle high-dimensional data and non-linear relationships. However,
to achieve optimal performance, it is crucial to tune the model's hyperparameters carefully.
Therefore, the Random Forest model can be an appropriate choice for classification tasks
with high-dimensional data, but the model's performance should be assessed and
hyperparameters should be optimized for best results.

### 6. Support Vector Machine (SVM):
Parameters used: C=0.1,kernel='rbf',probability=True

The Support Vector Machine (SVM) model was trained using cross-validation and grid
search to find the best hyperparameters. The evaluation metrics used to assess the model's
performance were sensitivity, specificity, accuracy, AUC score, and F1-score. The results
showed that the SVM model performed well, with high values for sensitivity, specificity,
accuracy, AUC score, and F1-score.
The grid search with a five-fold cross-validation strategy was used to find the best
hyperparameters for the SVM model. The best value for the regularization parameter (c) was
found to be 10, but a lower value of 0.1 was used instead to avoid overfitting and achieve the
highest accuracy of the model. The StratifiedKFold with three splits was used to ensure that
the distribution of the target variable was maintained in each fold.
In summary, the SVM model trained with cross-validation and grid search is a good choice
for classification tasks with high-dimensional data. The use of cross-validation and grid
search helps to optimize the model's hyperparameters and improve its performance. The
model accurately identifies positive and negative cases and effectively differentiates between
them.

Hyper Parameter Tuning:

The SVM model was optimized using cross-validation and grid search to find the best
hyperparameters. The regularization parameter (C) was tuned, and the best value was found to
be 10, but a lower value of 0.1 was used to avoid overfitting and achieve the highest accuracy
of the model. A five-fold cross-validation strategy was used during grid search, and
StratifiedKFold with three splits was used to maintain target variable distribution. The
optimized SVM model showed improved performance, with high sensitivity, specificity,
accuracy, AUC score, and F1-score.

## Project Results

![Image](https://github.com/anudeep-ga/Diabetes_Prediction/blob/main/Images/project_results.png?raw=true)
