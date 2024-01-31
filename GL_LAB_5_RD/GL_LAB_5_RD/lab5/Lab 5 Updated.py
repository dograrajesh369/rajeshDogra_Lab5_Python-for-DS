#!/usr/bin/env python
# coding: utf-8

# # <font color=darkblue> Machine Learning model deployment with Flask framework</font>

# ## <font color=Blue>Used Cars Price Prediction Application</font>

# ### Objective:
# 1. To build a Machine learning regression model to predict the selling price of the used cars based on the different input features like fuel_type, kms_driven, type of transmission etc.
# 2. Deploy the machine learning model with the help of the flask framework.

# ### Dataset Information:
# #### Dataset Source: https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho?select=CAR+DETAILS+FROM+CAR+DEKHO.csv
# This dataset contains information about used cars listed on www.cardekho.com
# - **Car_Name**: Name of the car
# - **Year**: Year of Purchase
# - **Selling Price (target)**: Selling price of the car in lakhs
# - **Present Price**: Present price of the car in lakhs
# - **Kms_Driven**: kilometers driven
# - **Fuel_Type**: Petrol/diesel/CNG
# - **Seller_Type**: Dealer or Indiviual
# - **Transmission**: Manual or Automatic
# - **Owner**: first, second or third owner
# 

# ### 1. Import required libraries

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score


import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# ### 2. Load the dataset

# In[2]:


df = pd.read_csv('D:/frontend/lab5/car_data.csv')
df.head()


# ### 3. Check the shape and basic information of the dataset.

# In[3]:


df.info()


# In[4]:


df.shape


# In[5]:


df.describe()


# In[6]:


df.describe(include='O')


# ### 4. Check for the presence of the duplicate records in the dataset? If present drop them

# In[7]:


## checking the presence of duplicate records
len(df[df.duplicated()])


# In[8]:


## dropping the duplicated record from the dataset
df.drop_duplicates(inplace=True)


# In[9]:


## recheck the presence of the duplicate record
len(df[df.duplicated()])


# ### 5. Drop the columns which you think redundant for the analysis.

# In[10]:


## dropping the redundant columns from the dataset.
df.drop(['Car_Name'],axis=1,inplace=True)


# ### 6. Extract a new feature called 'age_of_the_car' from the feature 'year' and drop the feature year

# In[11]:


df['age_of_the_car']=2024-df['Year']
df.drop('Year',axis=1, inplace=True)
df.head(5)


# ### 7. Encode the categorical columns

# In[12]:


df['Fuel_Type'].unique()


# In[13]:


df['Seller_Type'].unique()


# In[14]:


df['Transmission'].unique()


# In[15]:


df['Fuel_Type']=df['Fuel_Type'].replace({'Petrol':0 , 'Diesel' :1, 'CNG': 2})


# In[16]:


df['Seller_Type']=df['Seller_Type'].replace({'Dealer':0, 'Individual':1})


# In[17]:


df['Transmission']=df['Transmission'].replace({'Manual':0, 'Automatic':1})


# ### 8. Separate the target and independent features.

# In[18]:


X= df.drop('Selling_Price', axis =1)
Y= df['Selling_Price']


# ### 9. Split the data into train and test.

# In[19]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size= 0.3)

print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)


# ### 10. Build a Random forest Regressor model and check the r2-score for train and test.

# In[20]:


rf=RandomForestRegressor()
rf.fit(X_train,Y_train)


# In[21]:


Y_train_pred=rf.predict(X_train)
Y_test_pred=rf.predict(X_test)

r2_train=r2_score(Y_train,Y_train_pred)
r2_test=r2_score(Y_test,Y_test_pred)


# ### 11. Create a pickle file with an extension as .pkl

# In[26]:


import pickle
#saving model to disk
pickle.dump(rf,open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))


# ### 12. Create new folder/new project in visual studio/pycharm that should contain the "model.pkl" file *make sure you are using a virutal environment and install required packages.*

# ### a) Create a basic HTML form for the frontend

# Create a file **index.html** in the templates folder and copy the following code.

# In[ ]:





# ### b) Create app.py file and write the predict function

# In[ ]:





# ### 13. Run the app.py python file which will render to index html page then enter the input values and get the prediction.

# In[ ]:





# ### Happy Learning :)
