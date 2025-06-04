from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import pandas as pd


iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df["sepal_lw_ratio"] = df["sepal width (cm)"] / df["sepal length (cm)"]
df["petal_lw_ratio"] = df["petal width (cm)"] / df["petal length (cm)"]
y = df.target


x = df[df.columns.difference(['target'])]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

pipeline = Pipeline([
    ('pca', PCA(n_components=3)),
    ('scaler', StandardScaler()),
    ('clf', LinearSVC(C=0.5, penalty='l2', dual=False, max_iter=10000))
])

scores = cross_val_score(pipeline, x_train, y_train, cv=5)
print("Cross-validation scores:", scores)
print("Average CV score:", scores.mean())

def plot():
  df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
  plt.figure(figsize=(8, 6))
  plt.figure(figsize=(8, 6))
  for species in df['species'].unique():
      subset = df[df['species'] == species]
      plt.scatter(subset.iloc[:, 0], subset.iloc[:, 1], label=species)
  plt.show()
plot()

pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)
print("Accuracy score: ", accuracy_score(y_test, y_pred))

del pipeline, x_train, x_test, y_train, y_test
