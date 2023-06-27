#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

file1_df = pd.read_csv('patient_mesh_text_1.csv')


# In[45]:



# Filter the data for the year 2011
df_2011 = file1_df[file1_df["DATE_RECEIVED"].str.startswith("2017")]

# Count occurrences of the words "pain" and "hernia surgery" in the FOI_TEXT column
num_pain = df_2011["FOI_TEXT"].str.count("INCISIONAL").sum()
num_hernia_surgery1 = df_2011["FOI_TEXT"].str.count("INGUINAL").sum()
num_hernia_surgery2 = df_2011["FOI_TEXT"].str.count("UMBILICAL").sum()
num_hernia_surgery3 = df_2011["FOI_TEXT"].str.count("INCONTINENCE").sum()
num_hernia_surgery4 = df_2011["FOI_TEXT"].str.count("VAGINAL").sum()
num_hernia_surgery5 = df_2011["FOI_TEXT"].str.count("HERNIA SURGERY").sum()
num_hernia_surgery6 = df_2011["FOI_TEXT"].str.count("SUI|STRESS URINARY INCONTINENCE").sum()


# Print the results
print("Number of occurrences of 'INCISIONAL' in 2011:", num_pain)
print("Number of occurrences of 'INGUINAL' in 2011:", num_hernia_surgery1)
print("Number of occurrences of 'UMBILICAL' in 2011:", num_hernia_surgery2)
print("Number of occurrences of 'INCONTIENCE' in 2011:", num_hernia_surgery3)
print("Number of occurrences of 'VAGINAL' in 2011:", num_hernia_surgery4)
print("Number of occurrences of 'HERNIA SURGERY' in 2011:", num_hernia_surgery5)
print("Number of occurrences of 'SUI' in 2011:", num_hernia_surgery6)


# In[44]:


# Filter the data for the year 2011
df_2011 = file1_df[file1_df["DATE_RECEIVED"].str.startswith("2017")]
num_pain = df_2011["FOI_TEXT"].str.count("INCISIONAL SURGERY").sum()
num_hernia_surgery1 = df_2011["FOI_TEXT"].str.count("INGUINAL SURGERY").sum()
num_hernia_surgery2 = df_2011["FOI_TEXT"].str.count("UMBILICAL SURGERY").sum()
num_hernia_surgery3 = df_2011["FOI_TEXT"].str.count("INCONTINENCE").sum()
num_hernia_surgery4 = df_2011["FOI_TEXT"].str.count("VAGINAL SURGERY").sum()
num_hernia_surgery5 = df_2011["FOI_TEXT"].str.count("HERNIA SURGERY").sum()
num_hernia_surgery6 = df_2011["FOI_TEXT"].str.count("SUI|STRESS URINARY INCONTINENCE").sum()

# Print the results
print("Number of occurrences of 'INCISIONAL SURGERY' in 2011:", num_pain)
print("Number of occurrences of 'INGUINAL SURGERY' in 2011:", num_hernia_surgery1)
print("Number of occurrences of 'UMBILICAL SURGERY' in 2011:", num_hernia_surgery2)
print("Number of occurrences of 'INCONTIENCE' in 2011:", num_hernia_surgery3)
print("Number of occurrences of 'VAGINAL SURGERY' in 2011:", num_hernia_surgery4)
print("Number of occurrences of 'HERNIA SURGERY' in 2011:", num_hernia_surgery5)
print("Number of occurrences of 'SUI' in 2011:", num_hernia_surgery6)



# In[ ]:




