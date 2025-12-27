import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
print(type(data))
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print(df.sample(10), df.shape)
X,y=load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
p=Perceptron()
p.fit(X_train,y_train)
print(p.predict(X_test))
print(y_test)
