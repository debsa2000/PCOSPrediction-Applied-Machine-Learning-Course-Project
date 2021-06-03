import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv("C:/Users/Dayanand Saha/Documents/PCOS_Project/Data/data without infertility.csv")

# Deleting Nan values
del df['AMH(ng/mL)']
del df['Unnamed: 42']  # All NaN value
del df['Marraige Status (Yrs)']  # 1 NaN value
del df['Fast food (Y/N)']  # 1 NaN value

# Droppind PCOS (Y/N) column and creating target column y for predictions
X = df.drop(columns=["PCOS (Y/N)"])
y = df["PCOS (Y/N)"].values

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# Create kNN classifier
knn = KNeighborsClassifier(n_neighbors=2)  # Fit the classifier to the data
knn.fit(X_train, y_train)

#kNN accuracy and other measures
y_pred = knn.predict(X_test)
print("k Nearest Neighbors model accuracy(in %):", knn.score(X_test, y_test)*100)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Classification Report for kNN
predictions = knn.predict(X_test)
classi_report = classification_report(y_test, predictions)
print(classi_report)

# Plotting the confusion matrix for kNN
plt.subplots(figsize=(15, 5))
cf_matrix = confusion_matrix(y_test, predictions)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, annot_kws={'size': 15}, cmap='Pastel2')
plt.title("Confusion Matrix for kNN")
plt.show()
