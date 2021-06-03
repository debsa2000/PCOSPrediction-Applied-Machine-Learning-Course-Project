from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle

df = pd.read_csv('C:/Users/nicnnnn/Documents/PCOS_Project/Datasets/PCOS_withoutinfertility.csv')

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


#reading datasets
df_inf = pd.read_csv('C:/Users/nicnnnn/Documents/PCOS_Project/Datasets/PCOS_infertility.csv')
df_noinf = pd.read_csv('C:/Users/nicnnnn/Documents/PCOS_Project/Datasets/PCOS_withoutinfertility.csv')

#Sample data from df_inf
df_inf.sample(5)

#Sample data from df_noinf
df_noinf.sample(5)

### Feature Selection

#Identifying Features which have more than 0.40 correlation with PCOS(Y/N)
corr_features=df_noinf.corrwith(df_noinf["PCOS (Y/N)"]).abs().sort_values(ascending=False)
#features with correlation more than 0.4
corr_features=corr_features[corr_features>0.4].index
corr_features

df_inf.corrwith(df_inf["PCOS (Y/N)"]).abs()
df_noinf=df_noinf[corr_features]
df_noinf.head()

df_noinf.columns

plt.figure(figsize=(14,5))
plt.subplot(1,6,1)
sns.boxplot(x='PCOS (Y/N)',y='Follicle No. (R)',data=df_noinf)
plt.subplot(1,6,2)
sns.boxplot(x='PCOS (Y/N)',y='Follicle No. (L)',data=df_noinf)
plt.subplot(1,6,3)
sns.boxplot(x='PCOS (Y/N)',y='Skin darkening (Y/N)',data=df_noinf)
plt.subplot(1,6,4)
sns.boxplot(x='PCOS (Y/N)',y='hair growth(Y/N)',data=df_noinf)
plt.subplot(1,6,5)
sns.boxplot(x='PCOS (Y/N)',y='Weight gain(Y/N)',data=df_noinf)
plt.subplot(1,6,6)
sns.boxplot(x='PCOS (Y/N)',y='Cycle(R/I)',data=df_noinf)

plt.show()

#correlation matrix heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df_noinf.corr(), annot=True)
plt.show()
