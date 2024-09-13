#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# In[3]:


df = pd.read_csv("smoking.csv")
df =df.drop(columns =["ID","oral"])
df.head()


# In[3]:


df.shape
df.info()
df.describe()


# In[4]:


df.isnull().sum()


# In[5]:


sns.barplot(x=df["gender"],y=df["smoking"])
plt.show()


# In[7]:


sns.countplot(df["gender"],hue =df["smoking"])


# In[9]:


plt.figure(figsize=(10, 5))
df["smoking"].value_counts().plot.pie(autopct='%0.2f%%')
plt.show()


# In[11]:


plt.figure(figsize=(9,6))
sns.histplot(x=df['age'],hue=df["smoking"])
plt.show()


# In[12]:


for i in df.columns:
    if(df[i].dtype=='int64' or df[i].dtype=='float'):
        sns.boxplot(df[i])
        plt.show()


# In[8]:


from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
df["gender"]=le.fit_transform(df["gender"])
df["tartar"]=le.fit_transform(df["tartar"])
df["dental caries"]=le.fit_transform(df["dental caries"])


# In[7]:


X =df.iloc[:,:-1]
y =df["smoking"]
from sklearn.ensemble import ExtraTreesClassifier
model =ExtraTreesClassifier()
model.fit(X,y)
df1=pd.Series(model.feature_importances_,index =X.columns)
plt.figure(figsize=(8,8))
df1.nlargest(24).plot(kind='barh')
plt.show()


# In[20]:


X = df[["gender", "height(cm)", "Gtp", "hemoglobin", "triglyceride", "age", "weight(kg)", "waist(cm)", "HDL", "serum creatinine", "ALT", "fasting blood sugar", "relaxation", "LDL", "systolic"]]
y = df["smoking"]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)  

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train, y_train)  

y_pred = lr.predict(x_test)

from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


# In[21]:


from sklearn.tree import DecisionTreeClassifier
dt =DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
print(classification_report(y_test,y_pred))


# In[24]:


from sklearn.ensemble import BaggingClassifier
bagging_clf=BaggingClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=1000)
bagging_clf.fit(x_train,y_train).score(x_test,y_test)
y_pred=bagging_clf.predict(x_test)
print(classification_report(y_test,y_pred))


# In[27]:


from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
et = ExtraTreesClassifier(n_estimators=1000, random_state=42)
et.fit(x_train, y_train)
y_pred = et.predict(x_test)
print(classification_report(y_test, y_pred))


# In[26]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

rfc = RandomForestClassifier(n_estimators=1000, random_state=42)


rfc.fit(x_train, y_train)


y_pred = rfc.predict(x_test)


print(classification_report(y_test, y_pred))


# In[ ]:




