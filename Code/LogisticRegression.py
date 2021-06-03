import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Reading dataset
df = pd.read_csv('C:/Users/nicnnnn/Documents/PCOS_Project/Datasets/PCOS_withoutinfertility.csv')

# Treating errors in dataset
del df['AMH(ng/mL)']
del df['Unnamed: 42']  # All NaN value
del df['Marraige Status (Yrs)']  # 1 NaN value
del df['Fast food (Y/N)']  # 1 NaN value

# Dropping PCOS (Y/N) column and creating target column y for predictions
X = df.drop(columns=["PCOS (Y/N)"])
y = df["PCOS (Y/N)"]

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# Create LogisticRegression classifier
lrm=LogisticRegression()
lrm.fit(X_train,y_train)

# Prediction
y_pred=lrm.predict(X_test)

# Model Evaluation
print(f"Score in Test Data : {lrm.score(X_test,y_test)}")

cm=confusion_matrix(y_test, y_pred)
p_right=cm[0][0]+cm[1][1]
p_wrong=cm[0][1]+cm[1][0]

print(f"Right classification : {p_right}")
print(f"Wrong classification : {p_wrong}")

# LogisticRegression accuracy and other measures
print("Logistic Regression with feature selection accuracy(in %):", lrm.score(X_test, y_test)*100)

# Classification Report for LogisticRegression
classi_report = classification_report(y_test, y_pred)
print(classi_report)

# Plotting the confusion matrix for LogisticRegression
plt.subplots(figsize=(15, 5))
cf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, annot_kws={'size': 15}, cmap='Pastel2')
plt.title("Confusion Matrix for LogisticRegression")
plt.show()

# Plotting ROC curve with AUC
metrics.plot_roc_curve(lrm, X_test, y_test)
plt.show()

#Log Loss
from sklearn.metrics import log_loss
from matplotlib import pyplot
losses = [log_loss(y_test, [y for x in range(len(y_test))]) for y in y_pred]
# plot predictions vs loss
pyplot.plot(y_pred, losses)
pyplot.show()
print(losses)
