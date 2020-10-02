import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


df = pd.read_csv('IRIS.csv')
df.isna().sum()

sns.set()
sns.pairplot(df, hue='species', height=1.5);

x = df.drop(['species'], axis=1)
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.33, random_state=1)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

pred = knn.predict(x_test)

accuracy = knn.score(x_test,y_test)
accuracy

cm = confusion_matrix(y_test, pred)
cm

import pickle
with open('iris_data_model.pkl','wb') as f:
    pickle.dump(knn,f)



import joblib
joblib.dump(knn, 'iris_model.pkl')   

model = joblib.load(open('iris_model.pkl', 'rb'))
pred = [2.1, 4.1, 2.2, 4.3]
print(model.predict([pred]))
