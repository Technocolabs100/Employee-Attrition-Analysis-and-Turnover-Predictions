#!/usr/bin/env python
# coding: utf-8

# # Dataset:
# Acme Corporation has provided historical data on employee demographics, job satisfaction, work environment, performance metrics, and turnover status. This dataset spans the last five years and includes information on employees who have left the company and those who are still currently employed.
# 
# The dataset typically includes several features that provide insights into employee characteristics, job satisfaction, and performance. While the exact features may vary, here's a general list of common features you might find in such a dataset:
# 
# Dictionary
# Employee ID: A unique identifier for each employee.
# 
# Age: The age of the employee.
# 
# Attrition: A binary variable indicating whether the employee has left the company (1) or is still employed (0).
# 
# Business Travel: The frequency and nature of business-related travel (e.g., "Travel_Rarely," "Travel_Frequently," "Non-Travel").
# 
# Department: The department to which the employee belongs (e.g., "Sales," "Research & Development," "Human Resources").
# 
# Distance From Home: The distance of the employee's residence from the workplace.
# 
# Education: The employee's level of education (e.g., "1: 'Below College'," "2: 'College'," "3: 'Bachelor'," "4: 'Master'," "5: 'Doctor').
# 
# Education Field: The field in which the employee's education lies (e.g., "Life Sciences," "Medical," "Marketing").
# 
# Environment Satisfaction: The level of satisfaction with the work environment on a scale.
# 
# Gender: The gender of the employee.
# 
# Job Involvement: The degree to which the employee is involved in their job.
# 
# Job Level: The level or rank of the employee's position.
# 
# Job Role: The specific role or title of the employee's job.
# 
# Job Satisfaction: The level of satisfaction with the job on a scale.
# 
# Marital Status: The marital status of the employee.
# 
# Monthly Income: The monthly salary of the employee.
# 
# Num Companies Worked: The number of companies the employee has worked for.
# 
# Over Time: Whether the employee works overtime or not.
# 
# Performance Rating: The performance rating of the employee.
# 
# Relationship Satisfaction: The level of satisfaction with relationships at the workplace.
# 
# Stock Option Level: The level of stock options provided to the employee.
# 
# Total Working Years: The total number of years the employee has been working.
# 
# Training Times Last Year: The number of training sessions the employee attended last year.
# 
# Work-Life Balance: The balance between work and personal life.
# 
# Years At Company: The number of years the employee has been with the current company.
# 
# Years In Current Role: The number of years the employee has been in their current role.
# 
# Years Since Last Promotion: The number of years since the last time the employee was promoted.
# 
# Years With Current Manager: The number of years the employee has been working under the current manager.
# 
# Please note that this is a general list, and the actual dataset might include additional features or variations. It's essential to explore the dataset thoroughly to understand the specifics of each feature and its relevance to the analysis. 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


# # load the dataset

# In[2]:


df=pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[13]:


df.corr()


# In[9]:


df.isnull().sum().sort_values(ascending=False)


# In[10]:


df['Attrition'].value_counts()


# In[11]:


df['EducationField'].unique()


# In[12]:


df['Gender'].value_counts()


# In[14]:


df.columns


# In[15]:


df.duplicated()


# In[16]:


# Compute the correlation matrix
corr_matrix = df.corr()

# Plot the correlation matrix
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# In[17]:


df.skew()


# # visualisation of data

# In[18]:


df.hist(bins=30, figsize=(20, 15), color='blue', edgecolor='black')
plt.suptitle('Distribution of Numerical Features', y=1.02)
plt.show()


# In[19]:


sns.countplot(x='Attrition', data=df)
plt.title('Attrition Distribution')
plt.xlabel('Attrition')
plt.ylabel('Count')
plt.show()


# In[20]:


plt.figure(figsize = (10,5))

plt.title("Attrition by Gender and Education Field")
sns.countplot(x = 'Gender', hue = 'EducationField', data = df)


# In[21]:


plt.figure(figsize = (10,5))

plt.title("Attrition by Gender and Job Role")
sns.countplot(x = 'Gender', hue = 'JobRole', data = df)


# In[22]:


plt.figure(figsize = (10,5))

plt.title("Attrition by Gender Marital Status")
sns.countplot(x = 'Gender', hue = 'MaritalStatus', data = df)


# In[23]:


plt.figure(figsize = (10,5))

plt.title("Attrition by Education and Job Role")
sns.countplot(x = 'Education', hue = 'JobRole', data = df)


# In[24]:


plt.figure(figsize = (10,5))

plt.title("Attrition by Business Travel and Environment Satisfaction")
sns.countplot(x = 'BusinessTravel', hue = 'EnvironmentSatisfaction', data = df)


# In[25]:


# Show Attrition by Raise and Performance Rating
plt.figure(figsize = (10,5))


plt.title("Attrition by Raise and Performance Rating")
sns.countplot(x = 'PercentSalaryHike', hue = 'PerformanceRating', data = df)


# In[28]:


df.columns


# # insights from the visualisation
# 
# The attrition level is higher among male employees. Employees with education in life sciences and medicine leave the company at a higher rate than other professionals. Attrition levels are highest among men performing jobs as Sales Executives, Laboratory Technicians and Research Scientists. For women, attrition is highest at the Sales Executive level. Men and women are less likely to quit the Research Director or Manager job role. On a scale between 1 - 4 with four being the highest level of Environment Satisfaction, attrition was equal between levels 3 and 4.

# # Feature Engineering and model building

# In[27]:


label_encoder = LabelEncoder()
df['Attrition'] = label_encoder.fit_transform(df['Attrition'])

df = df.drop(columns=['EmployeeNumber', 'Over18', 'EmployeeCount', 'StandardHours'])

categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop('Attrition', axis=1)
y = df['Attrition']


# # Logistic Regression Model

# In[29]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return accuracy, precision, recall, f1

lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
lr_metrics = evaluate_model(lr, X_test, y_test)



# # random forest model

# In[30]:


rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_metrics = evaluate_model(rf, X_test, y_test)


# # Gradient Boosting Model

# In[31]:


gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
gb_metrics = evaluate_model(gb, X_test, y_test)


# In[32]:


print("Model Evaluation Metrics:")
print("Logistic Regression: Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}".format(*lr_metrics))
print("Random Forest: Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}".format(*rf_metrics))
print("Gradient Boosting: Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}".format(*gb_metrics))

print("\nClassification Reports:")
print("Logistic Regression:")
print(classification_report(y_test, lr.predict(X_test)))

print("Random Forest:")
print(classification_report(y_test, rf.predict(X_test)))

print("Gradient Boosting:")
print(classification_report(y_test, gb.predict(X_test)))


# # Insights:
# Overall Accuracy:
# 
# Logistic Regression: 86.05%
# Random Forest: 83.33%
# Gradient Boosting: 85.03%
# Logistic Regression has the highest overall accuracy, indicating it correctly classifies the majority of the instances. 

# In[ ]:




