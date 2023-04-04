import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
import requests
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


#QUESTION 1-2
url = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
response = requests.get(url)

with open('kc_house_data_NaN.csv', 'wb') as f:
    f.write(response.content)


df = pd.read_csv('kc_house_data_NaN.csv', index_col=0)

df = df.drop('id',axis=1)
df.describe()

df.head()

#QUESTION 3-4-5-6-7
#REMOVE NAN
mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)

mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)

new_data = df['floors'].value_counts().to_frame()
sns.boxplot(x=df['waterfront'],y=df['price'])


sns.regplot(x=df['sqft_above'],y=df['price'])
df.corr()['price'].sort_values(ascending=False)

X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)



lm.fit(df[['sqft_living']],df['price'])
yhat_a = lm.predict(df[['sqft_living']])
print(yhat_a)
lm.score(df[['sqft_living']],df['price'])


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
X = df[features]
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X,Y)


#QUESTION 8

W = df[features]
y = df['price']
Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe = Pipeline(Input)
pipe.fit(W,y)
pipe.score(W,y)

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)

print("training samples:",x_train.shape[0])
print("test samples:", x_test.shape[0])



#QUESTION 9

from sklearn.linear_model import Ridge
RDG_test = Ridge(alpha = 0.1)
RDG_test.fit(x_test, y_test)
RDG_test.score(x_test, y_test)



#QUESTION 10

pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)

RDG_test.fit(x_train_pr, y_train)
RDG_test.score(x_train_pr, y_train)