import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

df = pd.read_csv('C:/Users/nicnnnn/Documents/PCOS_Project/Datasets/PCOS_withoutinfertility.csv')

# Deleting redundant columns
del df["Unnamed: 42"]
del df['Patient File No.']
# print(df)

# Identifying entires with NULL (NaN)
null_columns = df.columns[df.isnull().any()]
df[null_columns].isnull().sum()
# print(df[df.isnull().any(axis=1)][null_columns])

# Deleting rows with NULL entries
df = df.dropna()
df.drop(df.index[305])
# print(df)

print(df.info())

# Checking datatype
# print(df["AMH(ng/mL)"].head())

# Converting to numeric value.
df["AMH(ng/mL)"] = pd.to_numeric(df["AMH(ng/mL)"], errors='coerce')

# Rechecking datatype
# print(df["AMH(ng/mL)"].head())

# # Generating a csv file that describes the Data
# # (Count, Mean, S.D, Min and Max for each column)
df.describe().T.to_csv("my_description.csv")


# Data Imbalance visualization plot
target = df['PCOS (Y/N)']
df.drop('PCOS (Y/N)', axis=1, inplace=True)
plt.figure(figsize=(8, 7))
sns.countplot(target)
plt.title('Data imbalance')
plt.show()

# Plotting Correlation Matrix
f = plt.figure(figsize=(19, 15))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=7, rotation=90)
plt.yticks(range(df.shape[1]), df.columns, fontsize=7)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.show()

# Plotting for outlier detection
# Excluding columns which has binary values
cols = list(df.columns)

for col in cols:
    if(col.strip()[-5:]) != "(Y/N)":
        plt.scatter([var for var in range(len(df[col]))], df[col], color='red')
        plt.xlabel('Sr. No.')
        plt.ylabel(col)
        plt.show()
