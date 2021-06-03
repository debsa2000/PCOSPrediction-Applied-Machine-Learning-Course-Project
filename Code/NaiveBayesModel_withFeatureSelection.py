import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Reading datasets
df_inf = pd.read_csv('C:/Users/nicnnnn/Documents/PCOS_Project/Datasets/PCOS_infertility.csv')
df_noinf = pd.read_csv('C:/Users/nicnnnn/Documents/PCOS_Project/Datasets/PCOS_withoutinfertility.csv')

# Sampling data from df_inf and df_noinf
df_inf.sample(5)
df_noinf.sample(5)

## Feature Selection
# Identifying Features which have more than 0.40 correlation with PCOS(Y/N)
corr_features=df_noinf.corrwith(df_noinf["PCOS (Y/N)"]).abs().sort_values(ascending=False)
# features with correlation more than 0.4
corr_features=corr_features[corr_features>0.4].index
corr_features
df_inf.corrwith(df_inf["PCOS (Y/N)"]).abs()
df_noinf=df_noinf[corr_features]
df_noinf.head()
df_noinf.columns

# Dropping PCOS (Y/N) column and creating target column y for predictions
X = df_noinf.drop(columns=["PCOS (Y/N)"])
y = df_noinf["PCOS (Y/N)"]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# Creating NaiveBayes Classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Prediction
y_pred=gnb.predict(X_test)

# Model Evaluation
print(f"Score in Test Data : {gnb.score(X_test,y_test)}")

cm=confusion_matrix(y_test, y_pred)
p_right=cm[0][0]+cm[1][1]
p_wrong=cm[0][1]+cm[1][0]

print(f"Right classification : {p_right}")
print(f"Wrong classification : {p_wrong}")

# Finding Naive Bayes accuracy and other measures
print("Naive Bayes with feature selection model accuracy(in %):", gnb.score(X_test, y_test)*100)
print('\nMean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Generating Classification Report for NaiveBayesClassifier
classi_report = classification_report(y_test, y_pred)
print(classi_report)

# Plotting the confusion matrix for NaiveBayesClassifier
plt.subplots(figsize=(15, 5))
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, annot_kws={'size': 15}, cmap='Pastel2')
plt.title("Confusion Matrix for Naive Bayes with feature selection")
plt.show()

# Plotting ROC curve with AUC
metrics.plot_roc_curve(gnb, X_test, y_test)
plt.show()
