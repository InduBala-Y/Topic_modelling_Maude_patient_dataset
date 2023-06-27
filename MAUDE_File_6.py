#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, pandas as pd

import matplotlib.pyplot as plt, seaborn as sns
import matplotlib as mpl

import math, string, re, pickle, json, time, os, sys, datetime, itertools

from tqdm.notebook import tqdm # This makes progress bars


# In[2]:


device_files = pd.DataFrame()
local = False

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
    for year in tqdm(range(2000, 2022)):
        if local:
            device_file = pd.read_csv('data/MAUDE_data/device%d.zip' % year)
        else:
            device_file = pd.read_csv('https://www.accessdata.fda.gov/MAUDE/ftparea/device%d.zip' % year, sep='|', quoting=3, encoding = "ISO-8859-1")
            device_file.to_csv('data/MAUDE_data/device%d.zip' % year, index = False)
        device_files = device_files.append(device_file)


# In[3]:


device_files["GENERIC_NAME"].value_counts().head(2)


# In[4]:


foitext_files = pd.DataFrame()
local = False

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
    for year in tqdm(range(2000, 2022)):
        if local:
            foitext_file = pd.read_csv('data/foitext%d.zip' % year)
        else:
            foitext_file = pd.read_csv('https://www.accessdata.fda.gov/MAUDE/ftparea/foitext%d.zip' % year, sep='|', quoting=3, encoding = "ISO-8859-1")
            foitext_file.to_csv('data/MAUDE_data/foitext%d.zip' % year, index = False)
            foitext_files = foitext_files.append(foitext_file)


# In[5]:


foitext_files.head(2)


# In[6]:


device_files = device_files[device_files['MDR_REPORT_KEY'].apply(type) == int] # Get only reports with an int key value
device_files = device_files.set_index('MDR_REPORT_KEY')
foitext_files = foitext_files.join(device_files, on = 'MDR_REPORT_KEY', how = 'left', rsuffix='_device')


# In[7]:


foitext_files.head(2)


# In[14]:


mesh2000_22 = foitext_files[foitext_files["GENERIC_NAME"].str.contains("MESH|mesh")== True]
mesh2000_22.head()


# In[15]:


# saving the dataframe
#mesh2000_22.to_csv('mesh2000_22.csv')


# In[16]:


mesh2000_22["GENERIC_NAME"].value_counts()


# In[18]:


mesh2000_22["GENERIC_NAME"].count()


# ### Loading device data

# In[4]:


mdrfoiThru2021_files = pd.DataFrame()
local = False

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
    for year in tqdm([2021]):
        if local:
            mdrfoi = pd.read_csv('data/MAUDE_data/mdrfoiThru%d.zip' % year)
        else:
            mdrfoi = pd.read_csv('https://www.accessdata.fda.gov/MAUDE/ftparea/mdrfoiThru%d.zip' % year, sep='|', quoting=3, encoding = "ISO-8859-1", on_bad_lines='skip')
            mdrfoi.to_csv('data/MAUDE_data/mdrfoiThru%d.zip' % year, index = False)
            mdrfoiThru2021_files = mdrfoiThru2021_files.append(mdrfoi)


# In[3]:


mesh2000_22 = pd.read_csv("mesh2000_22.csv")
mesh2000_22.head()


# In[6]:


mdrfoiThru2021_files.head()


# In[7]:


P_data= mdrfoiThru2021_files[mdrfoiThru2021_files["REPORT_SOURCE_CODE"] == 'P']
P_data.head(2)


# In[8]:


P_data["REPORT_SOURCE_CODE"].value_counts()


# In[11]:


# saving the dataframe
P_data.to_csv('pdata95-22.csv')


# In[ ]:





# In[12]:


# Matching reports

#mesh2000_22 = mesh2000_22[mesh2000_22['MDR_REPORT_KEY'].apply(type) == int] # Get only reports with an int key value
#patient_data = patient_data[patient_data['MDR_REPORT_KEY'].apply(type) == int]

mesh2000_22 = mesh2000_22.set_index('MDR_REPORT_KEY')


patient_mesh_df = P_data.join(mesh2000_22, on = 'MDR_REPORT_KEY', how = 'left', rsuffix='_device' )
                              


# In[14]:


patient_mesh_df


# In[16]:


patient_mesh_df['DATE_RECEIVED'] = pd.to_datetime(patient_mesh_df['DATE_RECEIVED'])


# In[17]:


patient_mesh_df['DATE_RECEIVED'] = pd.to_datetime(patient_mesh_df['DATE_RECEIVED'], format='%Y-%m-%d')


# In[19]:


patient_mesh_df = patient_mesh_df.loc[(patient_mesh_df["DATE_RECEIVED"] >= "2000-01-01")]
patient_mesh_df


# In[28]:


patient_mesh_df["FOI_TEXT"].value_counts()


# In[33]:


patient_mesh_df["FOI_TEXT"].isna().sum()


# In[34]:


patient_mesh_df.info()


# In[29]:


# saving the dataframe
patient_mesh_df.to_csv('patient_mesh2000_22.csv')


# In[45]:


patient_text = patient_mesh_df["FOI_TEXT"].dropna()
patient_text


# In[36]:


patient_mesh_df["ADVERSE_EVENT_FLAG"].value_counts()


# In[39]:


patient_mesh_df["PRODUCT_PROBLEM_FLAG"].value_counts()


# In[40]:


date_received = pd.to_datetime(patient_mesh_df['DATE_RECEIVED'], errors='coerce')
plt.hist(date_received, bins = 12*20)
plt.xlabel('Date Report Recieved')
plt.ylabel('Monthly Report Count')
plt.tight_layout()
plt.savefig('figs/FDA_report_counts.png', dpi = 300)


# In[41]:


patient_mesh_df.describe()


# In[47]:


mesh_text = ' '.join(str(review) for review in patient_text )
mesh_text


# In[48]:


patient_mesh_df["FOI_TEXT"].str.contains("PAIN|INFECTION|SICK|BLEEDING|EROSION|SEVERE|DEMAGE|TIGHT|HEALTH PROBLEM|ABNORMAL|ANXIETY|NEGATIVE|DIFFICULT|ACHES|PAINFUL|DIARRHEA|BOWEL OBSTRUCTION|INCONTINENCE|DIED|BOWEL PROBLEMS|ANAL|DYSPAREUNIA|PAINfUL SEXUAL INTERCOURSE|REMOVAL|COME OUT|WEAKNESS|NUMBNESS").value_counts()


# In[49]:


from wordcloud import WordCloud

word_cloud = WordCloud(collocations = False, background_color = 'white').generate(mesh_text)
# Display the generated Word Cloud
plt.imshow(word_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# In[ ]:




