from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle

df = pd.read_csv("C:/Users/Dayanand Saha/Documents/PCOS_Project/Data/data without infertility.csv")

# print(df.columns)
del df['AMH(ng/mL)']
del df['Unnamed: 42']  # All NaN value
del df['Marraige Status (Yrs)']  # 1 NaN value
del df['Fast food (Y/N)']  # 1 NaN value

X = df.drop(columns=["PCOS (Y/N)"])
y = df["PCOS (Y/N)"].values

# splitting data into trining and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=42)

# splitting training data into train and valid
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train, y_train, test_size=0.3, random_state=1, stratify=y_train)

# # Fitting the Gaussian Naive Bayes Model
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
#
# y_prob_gnb = gnb.predict_proba(X_valid)[:, 1]
# fpr_gnb, tpr_gnb, thresholds = roc_curve(y_valid, y_prob_gnb)
#
# # Plot ROC curve
# plt.figure(figsize=(8, 7))
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_gnb, tpr_gnb, label='Knn', color='c')
# plt.xlabel('fpr')
# plt.ylabel('tpr')
# plt.title('Gaussian Naive Bayes Classifier ROC curve')
# plt.show()
# # Print AUC
# auc_gnb = np.trapz(tpr_gnb, fpr_gnb)
# print('Gaussian Naive Bayes Classifier AUC:', auc_gnb)
#
# # Fitting the k nearest Neighbours Model
# knn = KNeighborsClassifier(n_neighbors=2)
# knn.fit(X_train, y_train)
#
# y_prob_knn = knn.predict_proba(X_valid)[:, 1]
# fpr_knn, tpr_knn, thresholds = roc_curve(y_valid, y_prob_knn)
#
# # Print ROC curve
# plt.figure(figsize=(8, 7))
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_knn, tpr_knn, label='Knn', color='c')
# plt.xlabel('fpr')
# plt.ylabel('tpr')
# plt.title('k Nearest Neighbours ROC curve')
# plt.show()
# # Print AUC
# auc_knn = np.trapz(tpr_knn, fpr_knn)
# print('k Nearest Neighbours AUC:', auc_knn)
#
# # Fitting the Random Forest model
# rfc = RandomForestClassifier()
# rfc.fit(X_train, y_train)
#
# y_prob_rfc = rfc.predict_proba(X_valid)[:, 1]
# fpr_rfc, tpr_rfc, thresholds = roc_curve(y_valid, y_prob_rfc)
#
# # Print ROC curve
# plt.figure(figsize=(8, 7))
# plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_rfc, tpr_rfc, color='c')
# plt.xlabel('fpr')
# plt.ylabel('tpr')
# plt.title('Random Forest Classifier ROC curve')
# plt.show()
# # Print AUC
# auc_rfc = np.trapz(tpr_rfc, fpr_rfc)
# print('Random Forest Classifier AUC:', auc_rfc)


def print_scores(m):
    res = [roc_auc_score(y_train, m.predict_proba(X_train)[:, 1]),
           roc_auc_score(y_valid, m.predict_proba(X_valid)[:, 1])]
    for r in res:
        print(r)


rf = RandomForestClassifier(n_jobs=-1, n_estimators=150, max_features='sqrt', min_samples_leaf=10)
rf.fit(X_train, y_train)

y_pred_proba = rf.predict_proba(X_valid)[:, 1]
fpr, tpr, thresholds = roc_curve(y_valid, y_pred_proba)


def get_fi(m, df):
    return pd.DataFrame({'col': df.columns, 'imp': m.feature_importances_}).sort_values('imp', ascending=False)


# lets get the feature importances for training set
fi = get_fi(rf, X_train)


def plot_fi(df):
    df.plot('col', 'imp', 'barh', figsize=(10, 10), color='mediumvioletred')


plot_fi(fi)

plt.show()
