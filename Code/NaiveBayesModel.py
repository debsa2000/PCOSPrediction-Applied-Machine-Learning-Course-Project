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

df = pd.read_csv("C:/Users/Dayanand Saha/Documents/PCOS_Project/Data/data without infertility.csv")

del df['AMH(ng/mL)']
del df['Unnamed: 42']  # All NaN value
del df['Marraige Status (Yrs)']  # 1 NaN value
del df['Fast food (Y/N)']  # 1 NaN value

# Droppind PCOS (Y/N) column and creating target column y for predictions
X = df.drop(columns=["PCOS (Y/N)"])
y = df["PCOS (Y/N)"].values

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Create NaiveBayes Classifier
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Naive Bayes accuracy and other measures
print("Naive Bayes model accuracy(in %):", gnb.score(X_test, y_test)*100)
y_pred = gnb.predict(X_test)
print('\nMean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Classification Report for NaiveBayesClassifier
predictions = gnb.predict(X_test)
classi_report = classification_report(y_test, predictions)
print(classi_report)

# Plotting the confusion matrix for NaiveBayesClassifier
plt.subplots(figsize=(15, 5))
cf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, annot_kws={'size': 15}, cmap='Pastel2')
plt.title("Confusion Matrix for Naive Bayes")
plt.show()
