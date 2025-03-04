#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis

# Importing library

# In[4]:


import pandas as pd


# Load the datasets 

# In[5]:


loan_status = pd.read_excel('Downloads//loan_status.xlsx')
loan_data = pd.read_excel('Downloads//loan_data.xlsx')
target_profit = pd.read_excel('Downloads//target_profit.xlsx')
loan_balances = pd.read_excel('Downloads//loan_balances.xlsx')
umbs_prices = pd.read_excel('Downloads//umbs_prices.xlsx')
loan_bids = pd.read_excel('Downloads//loan_bids.xlsx')


# Inspect the Data

# In[6]:


print(loan_status.head())
print(loan_data.head())
print(target_profit.head())
print(loan_balances.head())
print(umbs_prices.head())
print(loan_bids.head())


# Check the Missing Values 

# In[7]:


print(loan_status.isnull().sum())
print(loan_data.isnull().sum())
print(target_profit.isnull().sum())
print(loan_balances.isnull().sum())
print(umbs_prices.isnull().sum())
print(loan_bids.isnull().sum())


# In[8]:


print(loan_status.shape)
print(loan_data.shape)
print(target_profit.shape)
print(loan_balances.shape)
print(umbs_prices.shape)
print(loan_bids.shape)


# In[7]:


print(loan_status.info())
print(loan_data.info())
print(target_profit.info())
print(loan_balances.info())
print(umbs_prices.info())
print(loan_bids.info())


# Data Cleaning
# Handling missing values - example: fill missing numeric values with the mean

# In[9]:


loan_status.fillna(loan_status.mean(), inplace=True)
loan_data.fillna(loan_data.mean(), inplace=True)
target_profit.fillna(target_profit.mean(), inplace=True)
loan_balances.fillna(loan_balances.mean(), inplace=True)
umbs_prices.fillna(umbs_prices.mean(), inplace=True)
loan_bids.fillna(loan_bids.mean(), inplace=True)


# Ensure correct data types

# In[10]:


loan_balances['next_payment_due_date'] = pd.to_datetime(loan_balances['next_payment_due_date'])


# #Remove duplicate rows

# In[11]:


loan_status.drop_duplicates(inplace=True)
loan_data.drop_duplicates(inplace=True)
target_profit.drop_duplicates(inplace=True)
loan_balances.drop_duplicates(inplace=True)
umbs_prices.drop_duplicates(inplace=True)
loan_bids.drop_duplicates(inplace=True)


# #Create Loan Statuses
# #Define a function to determine loan status

# In[12]:


def determine_loan_status(row):
    if row['current_balance'] == 0:
        return 'Closed'
    elif row['next_payment_due_date'] < pd.Timestamp.today():
        return 'Delinquent'
    else:
        return 'Active'


# Apply the function to create the new column

# In[13]:


loan_balances['loan_status'] = loan_balances.apply(determine_loan_status, axis=1)


# Display the updated loan_balances DataFrame

# In[15]:


print(loan_balances.head())


# Amortize Loan Balances 

# In[16]:


def calculate_amortized_balance(principal, rate, n_periods):
    if rate == 0:
        return principal / n_periods
    rate_per_period = rate / 12
    amortized_balance = principal * (rate_per_period * (1 + rate_per_period) ** n_periods) / ((1 + rate_per_period) ** n_periods - 1)
    return amortized_balance

loan_balances['amortized_balance'] = loan_balances.apply(lambda row: calculate_amortized_balance(row['loan_amount'], row['interest_rate'], row['loan_term']), axis=1)


#  Merge Datasets 

# In[17]:


merged_df = loan_status.merge(loan_data, on='loan_id', how='left')\
    .merge(target_profit, on='loan_id', how='left')\
    .merge(loan_balances, on='loan_id', how='left')\
    .merge(loan_bids, on='loan_id', how='left')


#  Check for any discrepancies in the merged dataset 

# In[18]:


print(merged_df.info())


# In[19]:


merged_df.head()


# Save the Cleaned Data

# In[20]:


merged_df.to_csv('Downloads//cleaned_merged_loans.csv', index=False)


# In[ ]:




