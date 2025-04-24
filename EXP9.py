#!/usr/bin/env python
# coding: utf-8

# In[52]:


import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()


# In[53]:


data.info


# In[54]:


data.isnull().sum()


# In[55]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()


# In[56]:


x=data[["Position","Level"]]
x.head()


# In[57]:


y=data[["Salary"]]


# In[58]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[59]:


from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)


# In[60]:


from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse


# In[61]:


r2=metrics.r2_score(y_test,y_pred)
r2


# In[62]:


dt.predict([[5,6]])
print("Name:Hashwatha M")
print("Reg No:212223240051")


# In[ ]:




