# Data-Science-Process-on-Complex-Datasets
AIM
To Perform Data Science Process on a complex dataset and save the data to a file.

ALGORITHM
Step 1
Read the given Data

Step 2
Clean the Data Set using Data Cleaning Process

Step 3
Apply Feature Generation/Feature Selection Techniques on the data set

Step 4
Apply EDA /Data visualization techniques to all the features of the data set

CODE
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

df.head()

df.isnull().sum()

plt.figure(figsize=(5,5))

plt.title("Data with Outliers")

df.boxplot()

plt.show()

plt.figure(figsize=(5,5))

cols = ['size','tip','total_bill']

Q1 = df[cols].quantile(0.25)

Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.title("Dataset after removing outliers")

df.boxplot()

plt.show()

df['sex'].unique()

!pip install --upgrade category_encoders

from category_encoders import BinaryEncoder

be = BinaryEncoder()

data = be.fit_transform(df['sex'])

df = pd.concat([df,data],axis=1)

df

df['smoker'].unique()

data = be.fit_transform(df['smoker'])

df = pd.concat([df,data],axis=1)

df

df['day'].unique()

from sklearn.preprocessing import LabelEncoder,OrdinalEncoder

clim = ['Thur','Fri','Sat','Sun']

en= OrdinalEncoder(categories = [clim])

df['day']=en.fit_transform(df[["day"]])

df

df['time'].unique()

le = LabelEncoder()

df['time'] = le.fit_transform(df[["time"]])

df

df.drop('sex',axis=1,inplace=True)

df.drop('smoker',axis=1,inplace=True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

print("Min-max scaled data:")

print(scaled_data)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

print("Standard scaled data:")

print(scaled_data)

import seaborn as sns

sns.scatterplot(data=df)

sns.displot(df['size'],kde=True)

sns.scatterplot(x="total_bill", y="tip", data=df)

plt.title("Correlation between Tip Amount and Total Bill Amount")

plt.show()

df["tip_percent"] = df["tip"] / df["total_bill"]

sns.barplot(x=df['size'],y=df['tip_percent'],data=df)

plt.title("Tip Percentage by Dining Party Size")

plt.show()

sns.barplot(x=df['time'], y=df['total_bill'])

plt.title("Highest Total Bill Amount by Time")

plt.show()

df.corr()

sns.heatmap(df.corr(),annot=True)

OUTPUT
image image image image image image image image image image image image image

RESULT
Thus Data Science Process on a complex dataset was performed successfully.
