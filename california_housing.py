from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.svm import SVR
import pandas as pd

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)

df["Target"] = housing.target
df["BedrmsPerPerso"] = df.AveBedrms / df.AveOccup
cords = df[["Latitude", "Longitude"]]
pca = PCA(n_components=1)
df["Cords"] = pca.fit_transform(cords)

x = df[df.columns.difference(["Target", "Latitude", "Longitude", "AveOccup", "AveBedrms"])]

scaler = MinMaxScaler()
x = scaler.fit_transform(x)
y = df.Target.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

clf = RandomForestRegressor(n_estimators = 100, criterion = "squared_error", verbose = 1)
# clf = SVR(kernel = "rbf", verbose = True)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = r2_score(y_test, y_pred)
print(f"model r2_score: {accuracy}")
