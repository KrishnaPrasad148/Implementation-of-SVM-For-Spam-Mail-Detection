# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start
2.Load the email dataset containing messages and labels (spam or ham).
3.Preprocess the dataset (e.g., convert labels to binary form).
4.Convert text messages into numerical vectors using CountVectorizer.
5.Split the data into training and testing sets.
6.Create and train the SVM classifier using the training data.
7.Use the trained model to predict the labels on the test data.
8.Evaluate the model using accuracy and classification metrics.
9.Stop


## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by:  Krishna Prasad S
RegisterNumber:  212223230108
```
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

data = pd.read_csv("spam.csv", encoding='ISO-8859-1')
data.head()

data.info()

data.isnull().sum()

x = data["v2"].values
y = data["v1"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

cv = CountVectorizer()
x_train = cv.fit_transform(x_train)
x_test = cv.transform(x_test)

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)
print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

con = confusion_matrix(y_test, y_pred)
print("Confusion Matrix :\n", con)

cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)
```
## Output:
![SVM For Spam Mail Detection](sam.png)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
