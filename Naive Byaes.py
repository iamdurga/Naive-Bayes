#!/usr/bin/env python
# coding: utf-8

# 

# # Bayes Theorem
# 
# Naive Bayes work on the basis of `Bayes Theorem`
# 
# $$P(A\mid B)=\frac {P(B\mid A) \cdot P(A)}{P(B)}$$
# 
# Where, P(A/B) = Posterior Probability
#   
#        P(B/A) = Conditional Probability
#        
#        P(A) = Prior Probability
#        
#        P(B) = Marginal Probability. 
#        
#   Here, P(B) is constant for all values and hence does not contribute much to classifying the dataset, so we neglect that term while doing calculations.
# 
# 

# # Import Necessary Module

# In[3]:


import pandas as pd 


# # Data Load
# 
# Here we use diabetes data. If you are interested in obtaining a data set, please visit [link]. (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) Talking about the dataset, it predicts whether a patient has diabetes or not on the basis of measurements (pregnancys,glucosee,blood pressuree,skin thicknesss,insulinn, BMI,diabetes pedigree functionn,agee).

# In[4]:


df = pd.read_csv('diabetes.csv')
df.head()


# # Convert Continuous Data to Categorical Data 
# 
# For classification tasks, we always need data in categorical format, but our available data is in continuous type. So our first task is to convert the given data into categorical data while also dealing with missing values. To change the given data into categorical format, we use the "pandas cut function." which work in the following manner:
# 
# `cut(array, bins, level)`
# 
# If we do not provide the bins size it automatically create the bins size in this way,
# 
# M = maximum value of array
# 
# m = minimum value of array
# 
# R = M - m
# 
# bins = round(sqrt(R)) 
# 
# But, in our care, we already provided the bin size, so it must first find the maximum value and divide that value by the number of bins, and then categorize our dataset according to the level we have provided.

# In[5]:


labels = ['lower','medium','higher']
for i in df.columns[:-1]: 
    mean = df[i].mean() 
    df[i] = df[i].replace(0,mean) 
    df[i] = pd.cut(df[i],bins=len(labels),labels=labels)
    
    


# # Count Function
#  
#  The below count function simply counts how much data there is according to class and category, respectively.

# In[20]:


def count(df,colname,label,target):
    rule = (df[colname] == label) & (df['Outcome'] == target) 
    return len(df[rule])


# In[21]:


predicted = []


# In[22]:


probabilities = {0:{},1:{}}


# # Split Data into Train and Test Set
# 
# We use 70% of the data without any random process in this data splitting process; however, this is not a good approach because it may be affected by bias at times, so I strongly recommend you use the k-folds approach or any other random splitting process. 

# In[14]:


train_percent = 70
train_len = int((train_percent*len(df))/100)
train_X = df.iloc[:train_len,:]
test_X = df.iloc[train_len+1:,:-1]
test_y = df.iloc[train_len+1:,-1]


# # Prior probability
# 
# Now, let's calculate prior probability by using following formula,
# 
#  P(Outcome => 0) = Count(0)/Total Number of data
#  
# P(Outcome => 1) = Count(1)/Total Number of data 

# In[17]:


total_0 = count(train_X,'Outcome',0,0)
total_1 = count(train_X,'Outcome',1,1)
    
prior_prob_0 = total_0/len(train_X)
prior_prob_1 = total_1/len(train_X)


# # Conditional Probability
# In Bayes Theorem there is another kind of probability which is known as conditional probability. Lets calculate it using following formula.
# 
# Conditional-probability = P(Count of Category/Count of 0)
# 
# Conditional-probability = P(Count of Category/Count of 1)
# 

# In[18]:


for col in train_X.columns[:-1]:
        probabilities[0][col] = {}
        probabilities[1][col] = {}
        
        for category in labels:
            total_ct_0 = count(train_X,col,category,0)
            total_ct_1 = count(train_X,col,category,1)
            
            probabilities[0][col][category] = total_ct_0 / total_0
            probabilities[1][col][category] = total_ct_1 / total_1


# # Posterior probability
# 
# Finally, 
# 
# Posterior Probability = Conditional Probability * Prior probability

# In[19]:


for row in range(0,len(test_X)):
        prod_0 = prior_prob_0
        prod_1 = prior_prob_1
        for feature in test_X.columns:
            prod_0 *= probabilities[0][feature][test_X[feature].iloc[row]] 
            prod_1 *= probabilities[1][feature][test_X[feature].iloc[row]] 
            
        
        
        if prod_0 > prod_1:
            predicted.append(0)

        else:
            predicted.append(1)
            


# # Which is the class for the given dataset?

# In[11]:


have_diabetes = 0
do_not_have = 0

for i in predicted:
    if i == 0:
        do_not_have +=1 
    else:
        have_diabetes += 1
print(have_diabetes)
print(do_not_have)
print("Final predication for given dataset is patient do not have diabetes")


# # The Given Classification Model's Accuracy 

# In[12]:


tp,tn,fp,fn = 0,0,0,0
for j in range(0,len(predicted)):
    if predicted[j] == 0:
        #print(test_y.iloc[j])
        if test_y.iloc[j] == 0:
            tp += 1
        else:
            fp += 1
    else:
        if test_y.iloc[j] == 1:
            tn += 1
        else:
            fn += 1


# In[13]:


++print('Accuracy for training length '+str(train_percent)+'% : ',((tp+tn)/len(test_y))*100)

