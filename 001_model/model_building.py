import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('beer_dataset.csv')
X = df.drop('Popularity',axis=1)
y = df['Popularity']

steps = [('scaler', MinMaxScaler()),
         ('logr', LogisticRegression(class_weight='balanced'))]
model = Pipeline(steps)
model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
