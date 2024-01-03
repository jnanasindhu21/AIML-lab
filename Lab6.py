import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv('Lab6.csv')
print(data.head())

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

le_outlook = LabelEncoder()
X.loc[:, 'Outlook'] = le_outlook.fit_transform(X['Outlook'])

le_Temperature = LabelEncoder()
X.loc[:, 'Temperature'] = le_Temperature.fit_transform(X['Temperature'])

le_Humidity = LabelEncoder()
X.loc[:, 'Humidity'] = le_Humidity.fit_transform(X['Humidity'])

le_Windy = LabelEncoder()
X.loc[:, 'Windy'] = le_Windy.fit_transform(X['Windy'])

print(X.head())

le_PlayTennis = LabelEncoder()
y = le_PlayTennis.fit_transform(y)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

print("Accuracy is:", accuracy_score(classifier.predict(X_test), y_test))
