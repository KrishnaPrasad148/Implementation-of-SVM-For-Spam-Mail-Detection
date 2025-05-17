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

### DATASET:
![Screenshot 2025-05-17 173021](https://github.com/user-attachments/assets/c524fc5c-a354-43f4-8dbd-00e73b0e9bd4)

### DATASET INFO:
![Screenshot 2025-05-17 173030](https://github.com/user-attachments/assets/c9516aa1-427d-4604-8803-726edb42d362)

![Screenshot 2025-05-17 173037](https://github.com/user-attachments/assets/27ad6bf3-6d34-4116-a931-b25441bb0e38)

### Y_PRED:
![Screenshot 2025-05-17 173046](https://github.com/user-attachments/assets/cb596c37-18d0-435b-93b1-89ee7a656f4d)

### ACCURACY:
![Screenshot 2025-05-17 173051](https://github.com/user-attachments/assets/86452256-e8d0-441f-bfb6-56fd72916b3e)

### CONFUSION MATRIX:
![Screenshot 2025-05-17 173057](https://github.com/user-attachments/assets/32cff266-0627-4f83-859f-27a02f21bf52)

### CLASSIFICATION REPORT:
![Screenshot 2025-05-17 173104](https://github.com/user-attachments/assets/90d45f9f-8a8c-447b-abbb-91c340a18c21)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
