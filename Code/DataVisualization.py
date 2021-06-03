import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(
    "C:/Users/Dayanand Saha/Documents/PCOS_Project/Data/PCOS_data_without_infertility.csv")

# Plots to show relationship between numerical variable and one or more categorical variables
# Age and PCOS
age = sns.catplot(x='Age (yrs)', data=df, hue='PCOS (Y/N)', kind='count')
plt.title("Patients Condition by Age")
plt.show()

# Blood Group and PCOS
blood = sns.catplot(x='Blood Group', data=df, hue='PCOS (Y/N)', kind='count')
plt.title("Patients Blood Type")
plt.show()

# Marriage years and PCOS
marr_yrs = sns.catplot(x='Marraige Status (Yrs)', data=df, hue='PCOS (Y/N)', kind='count')
plt.title("Patients Years of Marriage")
plt.show()

# Weighted Correlation Heatmap
corrmat = df.corr()
corrmat["PCOS (Y/N)"].sort_values(ascending=False)
plt.figure(figsize=(12, 12))
k = 12  # number of variables with positive for heatmap
l = 3  # number of variables with negative for heatmap
cols_p = corrmat.nlargest(k, "PCOS (Y/N)")["PCOS (Y/N)"].index
cols_n = corrmat.nsmallest(l, "PCOS (Y/N)")["PCOS (Y/N)"].index
cols = cols_p.append(cols_n)

cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, cmap="BuPu", annot=True, square=True, fmt='.2f',
                 annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
