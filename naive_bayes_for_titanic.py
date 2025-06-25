import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = sns.load_dataset('titanic')
df = df.rename(columns={"sex": "gender"})

i = 0

algs = [CategoricalNB(), GaussianNB(), BernoulliNB()]
alg = algs[i]

norms = [StandardScaler(), OrdinalEncoder()]
norm = norms[1 if i == 0 else 0]

def encode_who(val: str):
  return ['manmale','womanfemale', 'childmale', 'childfemale'].index(val) + 1

def encode_class(val: str):
  return ['First','Second', 'Third'].index(val) +1

def encode_deck(val):
    return ['nan', 'A', 'B', 'C', 'D', 'E', 'F', 'G'].index(val) + 1


df["alone"] = df["alone"].apply(lambda x: int(x))
df = df.astype(str)

if i == 2:
  df2 = pd.get_dummies(df[["who", "class", "deck"]])
  df = pd.concat([df, df2], axis=1)
else:
  df["who"] = df["who"] + df["gender"]
  df["who"] = df["who"].apply(lambda x: encode_who(x))
  df["class"] = df["class"].apply(lambda x: encode_class(x))
  df['deck'] = df['deck'].astype(str)
  df["deck"] = df["deck"].apply(lambda x: encode_deck(x))
  df["age"] = df["age"].apply(lambda x: -9999 if x == "nan" else x)

if i == 2:
  df = df.drop(["embark_town", "alive", "adult_male", "gender", "pclass", "embarked", "who", "deck", "class", "age", "fare"], axis = 1) # <- BernoulliNB
elif i == 0:
  df = df.drop(["embark_town", "alive", "adult_male", "gender", "pclass", "embarked", "age", "fare"], axis = 1) # <- CategoricalNB & OrdinalEncoder
elif i == 1:
    df = df.drop(["embark_town", "alive", "adult_male", "gender", "pclass", "embarked"], axis = 1) # <- GaussianNB

x = np.array(df.drop("survived", axis  = 1).astype(float))
y = np.array(df["survived"].astype(int))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20)

pipeline = Pipeline([
    ('scaler', norm),
    ('clf', alg)
])

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

print(alg)
print(norm)
print(f"\naccuracy: {accuracy_score(y_pred, y_test)}")
df
