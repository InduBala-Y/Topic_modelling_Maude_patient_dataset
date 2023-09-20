#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np, pandas as pd

import matplotlib.pyplot as plt, seaborn as sns
import matplotlib as mpl

import math, string, re, pickle, json, time, os, sys, datetime, itertools

from tqdm.notebook import tqdm # This makes progress bars


# In[3]:


patientthru2022_files = pd.DataFrame()
local = False

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
    for year in tqdm([2022]):
        if local:
            patient = pd.read_csv('data/MAUDE_data/patientthru%d.zip' % year)
        else:
            patient = pd.read_csv('https://www.accessdata.fda.gov/MAUDE/ftparea/patientthru%d.zip' % year, sep='|', quoting=3, encoding = "ISO-8859-1", on_bad_lines='skip')
            patient.to_csv('data/MAUDE_data/patientthru%d.zip' % year, index = False)
            patientthru2022_files = patientthru2022_files.append(patient)


# In[4]:


patientthru2022_files


# In[6]:


patientthru2022_files.info()


# In[7]:


patientthru2022_files['DATE_RECEIVED'] = pd.to_datetime( patientthru2022_files['DATE_RECEIVED']
                                                       )


# In[8]:


patientthru2022_files['DATE_RECEIVED'] = pd.to_datetime( patientthru2022_files['DATE_RECEIVED'], format='%Y-%m-%d')


# In[12]:


patient_df1 = patientthru2022_files.loc[( patientthru2022_files["DATE_RECEIVED"] >= "2000-01-01") & (patientthru2022_files["DATE_RECEIVED"] < "2021-12-31")]
patient_df1


# In[28]:


patient_df1["DATE_RECEIVED"].value_counts()


# In[30]:


pmesh_df1 = pd.read_csv("final(2430).csv")
pmesh_df1


# In[31]:


pmesh_df1["DATE_RECEIVED"].value_counts()


# In[32]:


final_df1 = pmesh_df1.join(patient_df1, on = 'MDR_REPORT_KEY', how = 'inner', rsuffix='_device' )

final_df1


# In[33]:


final_df1["SEQUENCE_NUMBER_OUTCOME"].isnull().value_counts()


# In[35]:


# saving the dataframe
final_df1.to_csv('patient_mesh_text_1.csv')


# In[36]:


final_df1["DATE_RECEIVED"].value_counts()


# In[ ]:




