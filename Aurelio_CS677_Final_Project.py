#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##############################
# Rayhan Aurelio (U89109850) #
# CS 677 - Pinsky ############
# Final Project ##############
##############################


# In[ ]:


#####################
##### Libraries #####
#####################


# In[155]:


import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


#####################
##### Functions #####
#####################


# In[ ]:


####################
##### Datasets #####
####################


# In[ ]:


##########################
##### Pre-processing #####
##########################


# In[ ]:


# The original dataset was not delimited properly
# Therefore, the delimiter value was changed in Excel


# In[126]:


ratings = pd.read_csv("LasVegasTripDatasetDelimited.csv", error_bad_lines=False)
ratings.head(4)


# In[127]:


# Convert Yes/No to 1/0

ratings["Pool"] = np.where(ratings["Pool"] == "YES", 1, 0)
ratings["Gym"] = np.where(ratings["Gym"] == "YES", 1, 0)
ratings["Tennis court"] = np.where(ratings["Tennis court"] == "YES", 1, 0)
ratings["Spa"] = np.where(ratings["Spa"] == "YES", 1, 0)
ratings["Casino"] = np.where(ratings["Casino"] == "YES", 1, 0)
ratings["Free internet"] = np.where(ratings["Free internet"] == "YES", 1, 0)

ratings["Friends Trip"] = np.where(ratings["Traveler type"] == "Friends", 1, 0)
ratings["Business Trip"] = np.where(ratings["Traveler type"] == "Business", 1, 0)
ratings["Family Trip"] = np.where(ratings["Traveler type"] == "Families", 1, 0)
ratings["Solo Trip"] = np.where(ratings["Traveler type"] == "Solo", 1, 0)
ratings["Couple Trip"] = np.where(ratings["Traveler type"] == "Couples", 1, 0)

member_years = []

# Since some member years are negative, they are converted to 0.
for i in range(len(ratings["Member years"])):
    if(ratings["Member years"][i] < 0):
        member_years.append(0)
    else:
        member_years.append(ratings["Member years"][i])
        
ratings["Member years"] = member_years

ratings["Hotel stars"] = ratings["Hotel stars"].astype(float)

ratings["Start stay"] = ratings["Period of stay"].str.split("-", n = 1, expand = True)[[0]]
ratings["End stay"] = ratings["Period of stay"].str.split("-", n = 1, expand = True)[[1]]

start_stay_int = []
for i in range(len(ratings)):
    start_stay_int.append(datetime.datetime.strptime(ratings["Start stay"][i], "%b").month)

ratings["Start stay"] = start_stay_int

end_stay_int = []
for i in range(len(ratings)):
    end_stay_int.append(datetime.datetime.strptime(ratings["End stay"][i], "%b").month)

ratings["End stay"] = end_stay_int

ratings["Period of stay"] = ratings["End stay"] - ratings["Start stay"] + 1
ratings

period_of_stay_corrected = []

for i in range(len(ratings)):
    if(ratings["Period of stay"][i] < 0):
        period_of_stay_corrected.append(ratings["Period of stay"][i] + 12)
    else:
        period_of_stay_corrected.append(ratings["Period of stay"][i])

ratings["Period of stay"] = period_of_stay_corrected

ratings["Adjusted_Score"] = ratings["Helpful votes"] * ratings["Score"]

ratings_cleaned = ratings[["Hotel name", "Nr. reviews", "Nr. hotel reviews", "Helpful votes", "Friends Trip", "Business Trip", "Family Trip", "Solo Trip", "Couple Trip", "Pool", "Gym", "Tennis court", "Spa", "Casino", "Free internet", "Hotel stars", "Nr. rooms", "Member years", "Score", "Adjusted_Score"]]
ratings_cleaned


# In[ ]:


############################
##### Data Exploration #####
############################


# In[128]:


# Number of reviews = 504
len(ratings_cleaned)


# In[129]:


# List of variables
ratings_cleaned.columns

# Nr. reviews = Number of reviews the user has written on TripAdvisor
# Nr. hotel reviews = Number of hotel reviews the user has written on TripAdvisor
# Helpful votes = Number of "likes" recieved for that rating
# Friends Trip = Variable to know if the trip is friends in nature
# Business Trip = Variable to know if the trip is business in nature
# Family Trip = Variable to know if the trip is family in nature
# Solo Trip = Variable to know if the trip is solo in nature
# Couple Trip = Variable to know if the trip is couple in nature
# Pool = Variable if hotel has a pool
# Gym = Variable if hotel has a gym
# Tennis court = Variable if hotel has a tennis court
# Spa = Variable if hotel has a spa
# Casino = Variable if hotel has a casino
# Free internet = Variable if hotel offers free internet
# Hotel stars = How many stars the hotel has
# Nr. rooms = How many rooms the hotel has
# Member years = How long the user has been a member of TripAdvisor
# Score = The score the user gave (the dependent variable)
# Adjusted Score = The adjusted score multiplied by number of helpful votes


# In[130]:


# Names of hotel along with counts of reviews
ratings['Hotel name'].value_counts()


# In[131]:


Hotel_ratings_adjusted = ratings_cleaned.groupby('Hotel name').sum()
sorted_ratings_adjusted = Hotel_ratings_adjusted.sort_values('Adjusted_Score', ascending = False)
sorted_ratings_adjusted


# In[ ]:


# There are 24 reviews for every 21 hotel in this dataset


# In[132]:


# Where each review originates from
ratings['User country'].value_counts()


# In[133]:


ratings_cleaned.describe()


# In[134]:


ratings_cleaned.dtypes


# In[135]:


ratings_cleaned.hist(figsize=(32,32))


# In[83]:


sns.heatmap(ratings_cleaned.corr())


# In[101]:


ratings_cleaned.corr(method = 'pearson').Adjusted_Score.sort_values(ascending = False)


# In[ ]:


# By looking at the correlation
# How long a user has been on TripAdvisor affects the score recieved
# People going on business trips seem to leave higher ratings
# The favorite amenities are Pool, Tennis court, Free internet

# Families seems to leave the worst ratings, Gyms, Spas, and number of rooms 
# have a negative correlation


# In[38]:


######################
##### Regression #####
######################


# In[102]:


# Split dataset into training and testing
train, test = train_test_split(ratings_cleaned, test_size = 0.5)

train_data = train[["Nr. reviews", "Nr. hotel reviews", "Friends Trip", "Business Trip", "Family Trip", "Solo Trip", "Couple Trip", "Pool", "Gym", "Tennis court", "Spa", "Casino", "Free internet", "Hotel stars", "Nr. rooms", "Member years"]]
train_label = train[["Adjusted_Score"]]

test_data = test[["Nr. reviews", "Nr. hotel reviews", "Friends Trip", "Business Trip", "Family Trip", "Solo Trip", "Couple Trip", "Pool", "Gym", "Tennis court", "Spa", "Casino", "Free internet", "Hotel stars", "Nr. rooms", "Member years"]]
test_label = test[["Adjusted_Score"]]


# In[140]:


# Linear Regression
lin_reg = LinearRegression(fit_intercept=True)
lin_reg.fit(ratings_cleaned[["Nr. reviews", "Nr. hotel reviews", "Friends Trip", "Business Trip", "Family Trip", "Solo Trip", "Couple Trip", "Pool", "Gym", "Tennis court", "Spa", "Casino", "Free internet", "Hotel stars", "Nr. rooms", "Member years"]], ratings_cleaned[["Adjusted_Score"]])

print("Linear Regression Score: ",lin_reg.score(ratings_cleaned[["Nr. reviews", "Nr. hotel reviews", "Friends Trip", "Business Trip", "Family Trip", "Solo Trip", "Couple Trip", "Pool", "Gym", "Tennis court", "Spa", "Casino", "Free internet", "Hotel stars", "Nr. rooms", "Member years"]], ratings_cleaned[["Adjusted_Score"]]))

print("Linear Regression Intercept: ", lin_reg.intercept_)
print("Linear Regression Coefficient: ", lin_reg.coef_)


# In[158]:


# Linear Regression Train Test
lin_reg_train = LinearRegression(fit_intercept=True)
lin_reg_train.fit(train_data, train_label)

lin_reg_predict = lin_reg_train.predict(test_data)

# print(lin_reg_predict.tolist().asstr())
# print(test_label["Adjusted_Score"].tolist())

print("Linear Regression Split Score: ",lin_reg_train.score(test_data, test_label))

print("Linear Regression Split Intercept: ", lin_reg_train.intercept_)
print("Linear Regression Split Coefficient: ", lin_reg_train.coef_)

