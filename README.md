# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required packages and print the present data
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.  

## Program:
```python
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Deepak S
RegisterNumber: 212224230053
*/
import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Midhun/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
TOP 5 ELEMENTS

<img width="1454" height="233" alt="image" src="https://github.com/user-attachments/assets/1956c8cd-5b47-4cc4-8897-e745a9a709f3" />

<img width="1451" height="228" alt="image" src="https://github.com/user-attachments/assets/1c02610f-0b9a-423c-9c96-11ea3669c562" />


PRINT DATA
<img width="1448" height="475" alt="image" src="https://github.com/user-attachments/assets/f5fde3c4-d019-4b4d-b952-5e87c97ba404" />

CONFUSION ARRAY

<img width="1451" height="57" alt="image" src="https://github.com/user-attachments/assets/a361fbdd-37c8-447a-9f5b-86912415546b" />

ACCURACY VALUE

<img width="210" height="51" alt="image" src="https://github.com/user-attachments/assets/5a43ebd3-5518-42b2-a14b-ff59395d7aed" />

CLASSFICATION REPORT

<img width="1446" height="191" alt="image" src="https://github.com/user-attachments/assets/555571d3-5696-4586-bf36-dbdd64b90bd4" />

PREDICTION

<img width="303" height="33" alt="image" src="https://github.com/user-attachments/assets/d972b3cf-aa8e-4b1e-9a10-90cc5e6a6c08" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
