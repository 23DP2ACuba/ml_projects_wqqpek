import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np

df = sns.load_dataset('titanic')
df = df.rename(columns={"sex": "gender"})


def encode_who(val: str):
  return ['manmale','womanfemale', 'childmale', 'childfemale'].index(val) + 1

def encode_class(val: str):
  return ['First','Second', 'Third'].index(val) +1 

def encode_deck(val):
    return ['nan', 'A', 'B', 'C', 'D', 'E', 'F', 'G'].index(val) + 1


df["alone"] = df["alone"].apply(lambda x: int(x))
df = df.astype(str)
df["who"] = df["who"] + df["gender"]
df["who"] = df["who"].apply(lambda x: encode_who(x))
df["class"] = df["class"].apply(lambda x: encode_class(x))
df['deck'] = df['deck'].astype(str)
df["deck"] = df["deck"].apply(lambda x: encode_deck(x))
df["age"] = df["age"].apply(lambda x: -9999 if x == "nan" else x)
df = df.drop(["embark_town", "alive", "adult_male", "gender", "pclass", "embarked"], axis = 1)
x = np.array(df.drop("survived", axis  = 1).astype(float))
y = np.array(df["survived"].astype(int))

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', KMeans(n_clusters=2))
])

pipeline.fit(x)

def accuracy(x: np.array, y: np.array) -> float:
  score = 0
  for i in range(len(x)):
    inp = np.array(x[i].astype(float))
    inp = inp.reshape(-1, len(inp))
    y_act = y[i]
    y_pred = pipeline.predict(inp)
    score += y_act == y_pred
  return score/len(y)
print(f"accuracy: {accuracy(x, y)}")
df
