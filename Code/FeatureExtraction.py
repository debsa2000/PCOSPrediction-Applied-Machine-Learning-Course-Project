import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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

# Performing standard scalar normalization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Applying PCA
pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

explained_variance = pca.explained_variance_ratio_
pca = PCA(n_components=1)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# RandomForestClassifier use
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Results and Confusion matrix
print('Accuracy=')
print(accuracy_score(y_test, y_pred))
plt.subplots(figsize=(15, 5))
cf_matrix= confusion_matrix(y_test, y_pred)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, annot_kws={'size': 15}, cmap='Pastel2')
plt.title("Confusion Matrix for PCA")
plt.show()
