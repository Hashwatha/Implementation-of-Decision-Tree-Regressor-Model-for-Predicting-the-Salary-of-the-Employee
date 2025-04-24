# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. Calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Hashwatha M
RegisterNumber: 212223240051
*/
```
```
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
print("Name:Hashwatha M")
print("Reg No:212223240051")

```
## Output:
## Data Head:
![image](https://github.com/user-attachments/assets/deb4c8d6-4c34-4e11-8cde-19b095269d00)
## Data Info:
![image](https://github.com/user-attachments/assets/462add9f-948d-4edc-a344-03aade24669a)
## isnull() sum():
![image](https://github.com/user-attachments/assets/73087e4c-c41b-4c64-8053-3df0dcfd3688)
## Data Head for salary:
![image](https://github.com/user-attachments/assets/ef5dc306-97c9-4dad-979d-7f31eda98c35)
## Mean Squared Error :
![image](https://github.com/user-attachments/assets/d83e0dbd-d3e9-4cf5-a73e-fc7f09f73b59)
## r2 Value:
![image](https://github.com/user-attachments/assets/7662e923-5df4-45b3-ad2c-260c0682ec71)
## Data prediction :
![image](https://github.com/user-attachments/assets/4ac72c0d-d254-4465-b4b5-389909dcec01)

![image](https://github.com/user-attachments/assets/19652dbd-ab9b-4d48-ae5a-f703f6cef57f)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
