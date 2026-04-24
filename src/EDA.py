import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier 



df = pd.read_csv("./data/customer_info.csv", sep = ',')

pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# print(df.head())


# print(df.info())
# print(df.describe())

print(df.isna().sum()) 

df = df.drop_duplicates(subset='customer_id', keep='first')

print(df.duplicated().sum()) 

scaler = StandardScaler()




