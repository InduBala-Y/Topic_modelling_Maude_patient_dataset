#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, pandas as pd

import matplotlib.pyplot as plt, seaborn as sns
import matplotlib as mpl

import math, string, re, pickle, json, time, os, sys, datetime, itertools

from tqdm.notebook import tqdm # This makes progress bars
from dask import dataframe as dd


# ### Loading device data

# In[3]:


patient_mesh_df = pd.read_csv("patient_mesh2000_22.csv" )
patient_mesh_df


# In[5]:


patient_mesh_df.columns.values


# In[23]:


new_df = patient_mesh_df[["MDR_REPORT_KEY", "REPORT_SOURCE_CODE","MANUFACTURER_LINK_FLAG_",
                       "DATE_RECEIVED","REPROCESSED_AND_REUSED_FLAG", "ADVERSE_EVENT_FLAG", "DATE_RETURNED_TO_MANUFACTURER",
                         "PRODUCT_PROBLEM_FLAG", "FOI_TEXT","MDR_TEXT_KEY",
                        "TEXT_TYPE_CODE", "GENERIC_NAME", "SUMMARY_REPORT",
                         "EVENT_TYPE" ]]
new_df


# In[38]:


new_df.groupby("PRODUCT_PROBLEM_FLAG")["GENERIC_NAME"].count().plot()
plt.show()


# In[47]:


adverse_flag = new_df[new_df["ADVERSE_EVENT_FLAG"] == "Y"]


# In[46]:


new_df["DATE_RECEIVED"] = pd.to_datetime(new_df["DATE_RECEIVED"])


# In[49]:


#adverse_flag["Year"] = 
adverse_flag["Year"] = (adverse_flag["DATE_RECEIVED"]).dt.year
adverse_flag


# In[13]:


new_df["ADVERSE_EVENT_FLAG"].value_counts()


# In[15]:


new_df["PRODUCT_PROBLEM_FLAG"].value_counts()


# In[101]:


adverse_flag.groupby("Year")["ADVERSE_EVENT_FLAG"].count()


# In[ ]:


1030, 813,898,837,863,838,1509,1763,2034,1899, 1471,1797,1963,2334,3284,5794,4607,3232,3562,4428,2641,2991


# In[98]:


plt.figure(figsize = (10, 8))
plt.xticks(range(2000,2022, 2))
adverse_flag.groupby("Year")["ADVERSE_EVENT_FLAG"].count().plot()

plt.title("Reported adverse events")
plt.xlabel("Years")
plt.ylabel("No. of Reports")
plt.show()


# In[102]:


adverse_flag["PRODUCT_PROBLEM_FLAG"].value_counts()


# In[110]:


adverse_flag2 = new_df[new_df["PRODUCT_PROBLEM_FLAG"] == "Y"]


# In[111]:


adverse_flag2["Year"] = (adverse_flag2["DATE_RECEIVED"]).dt.year
adverse_flag2.head()


# In[106]:


adverse_flag2["PRODUCT_PROBLEM_FLAG"].value_counts()


# In[112]:


adverse_flag2.groupby("Year")["PRODUCT_PROBLEM_FLAG"].count()


# In[125]:


plt.figure(figsize = (10, 8))
plt.xticks(range(2000,2022, 2))

x_axis = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
         2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]           
y_axis = [2386, 2484, 2675, 2711, 2815, 2668, 2858, 3028, 3016, 3357, 3436, 3547, 3281, 3408, 
           3021, 3161, 3676, 4278, 5255, 6906, 5608, 6816]
#adverse_flag.groupby("Year")["PRODUCT_PROBLEM_FLAG"].count().plot()
plt.plot(x_axis, y_axis)
plt.title("Reported Device(Product) Problem")
plt.xlabel("Years")
plt.ylabel("No. of Reports")
plt.show()


# In[3]:


plt.figure(figsize = (10, 8))
plt.xticks(range(2000,2022, 2))

x_axis = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
         2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021]


y = [1030, 813,898,837,863,838,1509,1763,2034,1899, 1471,1797,1963,2334,3284,5794,4607,3232,3562,4428,2641,2991]
z = [2386, 2484, 2675, 2711, 2815, 2668, 2858, 3028, 3016, 3357, 3436, 3547, 3281, 3408, 
           3021, 3161, 3676, 4278, 5255, 6906, 5608, 6816]

plt.plot(x_axis, y, color = "r", label ="Reported adeverse event outcome")
plt.plot(x_axis, z, color = "g", label ="Reported adeverse device outcome")
plt.title("Reported adverse outcomes")
plt.legend()
plt.xlabel("Years")
plt.ylabel("No. of Reports")
plt.show()



# In[79]:


adverse_flag["GENERIC_NAME"].value_counts().head(50)


# In[76]:


adverse_flag["GENERIC_NAME"].value_counts().head(10).plot()
plt.show()
plt()


# In[81]:


adverse_flag["GENERIC_NAME"].count()


# In[2]:


x_axis = [ "Mesh Surgical Urogynecologic ", "Pelvic Mesh", "Other Mesh"]
y_axis = [987, 512, 376]
plt.figure(figsize = (10, 8))
plt.bar(x_axis, y_axis, width=0.4)

plt.title("Adverse event Vs Mesh")
plt.ylabel("No. of Mesh used")
plt.show()


# In[61]:


adverse_flag.groupby("GENERIC_NAME")["ADVERSE_EVENT_FLAG"].count()


# In[69]:


ad = adverse_flag.groupby("GENERIC_NAME")
ad.sum()


# In[62]:


adverse_flag.groupby("GENERIC_NAME")["ADVERSE_EVENT_FLAG"].count().plot()
plt.title("Reported adverse events")
plt.xlabel("Years")
plt.ylabel("No. of Reports")
plt.show()


# In[ ]:





# In[ ]:




