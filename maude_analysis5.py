#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np, pandas as pd

import matplotlib.pyplot as plt, seaborn as sns
import matplotlib as mpl

import math, string, re, pickle, json, time, os, sys, datetime, itertools

from tqdm.notebook import tqdm # This makes progress bars


# In[24]:


file1_df = pd.read_csv("patient_mesh_text_1.csv")
file1_df.head(5)


# In[3]:


file1_df["DATE_RECEIVED"] = pd.to_datetime(file1_df["DATE_RECEIVED"])


# In[4]:


file1_df["Year"] = (file1_df["DATE_RECEIVED"]).dt.year
file1_df


# In[9]:


import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt


# Convert the DATE_RECEIVED column to datetime format
file1_df['DATE_RECEIVED'] = pd.to_datetime(file1_df['DATE_RECEIVED'], format='%Y-%m-%d')

# Perform sentiment analysis using the VADER lexicon
sia = SentimentIntensityAnalyzer()
file1_df['sentiment'] = file1_df['FOI_TEXT'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Create a year column based on the DATE_RECEIVED column
file1_df['year'] = file1_df['DATE_RECEIVED'].dt.year

# Group the data by year and sentiment and count the number of reviews in each group
grouped_data = file1_df.groupby(['year', 'sentiment'])['FOI_TEXT'].count().reset_index()

# Pivot the data to create a matrix with years as rows, sentiment as columns, and review counts as values
pivoted_data = grouped_data.pivot(index='year', columns='sentiment', values='FOI_TEXT').fillna(0)


# In[11]:


# Perform K-Means clustering on the sentiment scores, with 3 clusters (negative, neutral, positive)
kmeans = KMeans(n_clusters=3, random_state=0).fit(file1_df[['sentiment']])

# Add a cluster column to the data
file1_df['cluster'] = kmeans.labels_

# Group the data by year and cluster and count the number of reviews in each group
grouped_data = file1_df.groupby(['year', 'cluster'])['FOI_TEXT'].count().reset_index()

# Pivot the data to create a matrix with years as rows, clusters as columns, and review counts as values
pivoted_data = grouped_data.pivot(index='year', columns='cluster', values='FOI_TEXT').fillna(0)






# Create a stacked bar chart of the pivoted data
colors = ['r', 'gray', 'g']
labels = ['Negative', 'Neutral', 'Positive']
pivoted_data.plot(kind='bar', stacked=True, color=colors)
plt.title('Sentiment Clustering by Year')
plt.xlabel('Year')
plt.ylabel('Number of Reviews')
plt.legend(labels=labels, loc='upper left')
plt.show()


# In[33]:


file1_df.columns.values


# In[9]:


import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt


# Convert the DATE_RECEIVED column to datetime format
file1_df['DATE_RECEIVED'] = pd.to_datetime(file1_df['DATE_RECEIVED'], format='%Y-%m-%d')

# Perform sentiment analysis using the VADER lexicon
sia = SentimentIntensityAnalyzer()
file1_df['sentiment'] = file1_df['FOI_TEXT'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Create a year column based on the DATE_RECEIVED column
file1_df['year'] = file1_df['DATE_RECEIVED'].dt.year

# Group the data by year and sentiment and count the number of reviews in each group
grouped_data = file1_df.groupby(['year', 'sentiment'])['FOI_TEXT'].count().reset_index()

# Pivot the data to create a matrix with years as rows, sentiment as columns, and review counts as values
pivoted_data = grouped_data.pivot(index='year', columns='sentiment', values='FOI_TEXT').fillna(0)

# Create a stacked bar chart of the pivoted data
pivoted_data.plot(kind='bar', stacked=True)
plt.title('Sentiment Analysis by Year')
plt.xlabel('Year')
plt.ylabel('Number of Reviews')
plt.show()


# In[10]:


import pandas as pd
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt




# Perform K-Means clustering on the sentiment scores, with 3 clusters (negative, neutral, positive)
kmeans = KMeans(n_clusters=3, random_state=0).fit(file1_df[['sentiment']])

# Add a cluster column to the data
file1_df['cluster'] = kmeans.labels_

# Group the data by year and cluster and count the number of reviews in each group
grouped_data = file1_df.groupby(['year', 'cluster'])['FOI_TEXT'].count().reset_index()

# Pivot the data to create a matrix with years as rows, clusters as columns, and review counts as values
pivoted_data = grouped_data.pivot(index='year', columns='cluster', values='FOI_TEXT').fillna(0)






# Create a stacked bar chart of the pivoted data
colors = ['r', 'gray', 'g']
labels = ['Negative', 'Neutral', 'Positive']
pivoted_data.plot(kind='bar', stacked=True, color=colors)
plt.title('Sentiment Clustering by Year')
plt.xlabel('Year')
plt.ylabel('Number of Reviews')
plt.legend(labels=labels, loc='upper left')
plt.show()


# In[15]:


import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Convert the DATE_RECEIVED column to datetime format
file1_df['DATE_RECEIVED'] = pd.to_datetime(file1_df['DATE_RECEIVED'], format='%Y-%m-%d')

# Perform sentiment analysis using the VADER lexicon
sia = SentimentIntensityAnalyzer()
file1_df['sentiment'] = file1_df['FOI_TEXT'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Assign sentiment labels based on the sentiment scores
file1_df['sentiment_label'] = file1_df['sentiment'].apply(lambda x: 'Negative' if x < 0 else 'Positive' if x > 0.1 else 'Neutral')

# Create a year column based on the DATE_RECEIVED column
file1_df['year'] = file1_df['DATE_RECEIVED'].dt.year

# Perform K-Means clustering on the sentiment scores, with 3 clusters (negative, neutral, positive)
kmeans = KMeans(n_clusters=3, random_state=0).fit(file1_df[['sentiment']])

# Add a cluster column to the data
file1_df['cluster'] = kmeans.labels_

# Group the data by year and cluster and count the number of reviews in each group
grouped_data = file1_df.groupby(['year', 'cluster', 'sentiment_label'])['FOI_TEXT'].count().reset_index()

# Pivot the data to create a matrix with years as rows, clusters as columns, and review counts as values
pivoted_data = grouped_data.pivot(index='year', columns=['cluster', 'sentiment_label'], values='FOI_TEXT').fillna(0)

# Create a stacked bar chart of the pivoted data
colors = ['r', 'gray', 'g']
labels = ['Negative', 'Neutral', 'Positive']
pivoted_data.plot(kind='bar', stacked=True, color=colors)
plt.title('Sentiment Clustering by Year')
plt.xlabel('Year')
plt.ylabel('Number of Reviews')
plt.legend(labels=labels, loc='upper left')
plt.show()


# In[25]:


import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Convert the DATE_RECEIVED column to datetime format
file1_df['DATE_RECEIVED'] = pd.to_datetime(file1_df['DATE_RECEIVED'], format='%Y-%m-%d')

# Perform sentiment analysis using the VADER lexicon
sia = SentimentIntensityAnalyzer()
file1_df['sentiment'] = file1_df['FOI_TEXT'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Assign sentiment labels based on the sentiment scores
file1_df['sentiment_label'] = file1_df['sentiment'].apply(lambda x: 'Negative' if x < 0 else 'Positive' if x > 0.1 else 'Neutral')

# Create a year column based on the DATE_RECEIVED column
file1_df['year'] = file1_df['DATE_RECEIVED'].dt.year

# Perform K-Means clustering on the sentiment scores, with 3 clusters (negative, neutral, positive)
kmeans = KMeans(n_clusters=3, random_state=0).fit(file1_df[['sentiment']])

# Add a cluster column to the data
file1_df['cluster'] = kmeans.labels_

# Group the data by year and cluster and count the number of reviews in each group
grouped_data = file1_df.groupby(['year', 'cluster', 'sentiment_label'])['FOI_TEXT'].count().reset_index()

# Pivot the data to create a matrix with years as rows, clusters as columns, and review counts as values
pivoted_data = grouped_data.pivot(index='year', columns=['cluster', 'sentiment_label'], values='FOI_TEXT').fillna(0)

# Add percentage columns
pivoted_data['Total'] = pivoted_data.sum(axis=1)
pivoted_data['Negative %'] = pivoted_data['Negative'] / pivoted_data['Total'] * 100
pivoted_data['Neutral %'] = pivoted_data['Neutral'] / pivoted_data['Total'] * 100
pivoted_data['Positive %'] = pivoted_data['Positive'] / pivoted_data['Total'] * 100

# Create a stacked bar chart of the pivoted data
colors = ['r', 'gray', 'g']
labels = ['Negative', 'Neutral', 'Positive']
pivoted_data[['Negative %', 'Neutral %', 'Positive %']].plot(kind='bar', stacked=True, color=colors)
plt.title('Sentiment Clustering by Year')
plt.xlabel('Year')
plt.ylabel('Percentage of Reviews')
plt.legend(labels=labels, loc='upper left')
plt.show()


# In[21]:


import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Convert the DATE_RECEIVED column to datetime format
file1_df['DATE_RECEIVED'] = pd.to_datetime(file1_df['DATE_RECEIVED'], format='%Y-%m-%d')

# Perform sentiment analysis using the VADER lexicon
sia = SentimentIntensityAnalyzer()
file1_df['sentiment'] = file1_df['FOI_TEXT'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Assign sentiment labels based on the sentiment scores
file1_df['sentiment_label'] = file1_df['sentiment'].apply(lambda x: 'Negative' if x < 0 else ('Positive' if x > 0.1 else 'Neutral'))

# Create a year column based on the DATE_RECEIVED column
file1_df['year'] = file1_df['DATE_RECEIVED'].dt.year

# Perform K-Means clustering on the sentiment scores, with 3 clusters (negative, neutral, positive)
kmeans = KMeans(n_clusters=3, random_state=0).fit(file1_df[['sentiment']])

# Add a cluster column to the data
file1_df['cluster'] = kmeans.labels_

# Group the data by year and cluster and sum the number of reviews in each group
grouped_data = file1_df.groupby(['year', 'cluster', 'sentiment_label'])['FOI_TEXT'].count().reset_index()

# Add a column for total reviews in each year
grouped_data['total_reviews'] = grouped_data.groupby('year')['FOI_TEXT'].transform('sum')

# Calculate the percentage of each sentiment label for each year
grouped_data['percentage'] = grouped_data['FOI_TEXT'] / grouped_data['total_reviews'] * 100

# Pivot the data to create a matrix with years as rows, clusters as columns, and review percentages as values
pivoted_data = grouped_data.pivot(index='year', columns=['cluster', 'sentiment_label'], values='percentage').fillna(0)

# Create a stacked bar chart of the pivoted data
colors = ['r', 'gray', 'g']
labels = ['Negative', 'Neutral', 'Positive']
pivoted_data.plot(kind='bar', stacked=True, color=colors)
plt.title('Sentiment Clustering by Year')
plt.xlabel('Year')
plt.ylabel('Percentage of Reviews')
plt.ylim([0,100])
plt.yticks(range(0, 101, 10))
plt.legend(labels)
plt.show()


# In[31]:



# Filter the data by the year 2011
file1_2011 = file1_df[file1_df['year'] == 2011]

# Perform t-SNE clustering on the sentiment scores, with 3 clusters (negative, neutral, positive)
tsne = TSNE(n_components=2, random_state=0).fit_transform(file1_2011[['sentiment']])
file1_2011['x'] = tsne[:, 0]
file1_2011['y'] = tsne[:, 1]

# Add a cluster column to the data
file1_2011['cluster'] = file1_2011['sentiment'].apply(lambda x: 0 if x < -0.05 else 1 if x > 0.05 else 2)

# Create a color palette with 3 colors for the 3 clusters
palette = sns.color_palette("colorblind", 3)

# Create a scatterplot of the data, with points colored by cluster
sns.scatterplot(data=file1_2011, x="x", y="y", hue="cluster", palette=palette)

# Set the legend to display 3 sentiment categories
plt.legend(title="Sentiment", labels=["Negative", "Neutral", "Positive"])

# Set the title of the plot
plt.title("2011 Sentiment Clusters")

plt.show()


# In[13]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt

# Convert the DATE_RECEIVED column to datetime format
file1_df['DATE_RECEIVED'] = pd.to_datetime(file1_df['DATE_RECEIVED'], format='%Y-%m-%d')

# Tokenize the FOI_TEXT column
stop_words = set(stopwords.words('english'))
file1_df['tokens'] = file1_df['FOI_TEXT'].apply(lambda x: [word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words])

# Extract keywords from the tokens column
keywords = Counter()
for token_list in file1_df['tokens']:
    keywords.update(token_list)
    
# Select the top 10 keywords and create a bar chart
top_keywords = dict(keywords.most_common(10))
plt.bar(top_keywords.keys(), top_keywords.values())
plt.title('Top 10 Keywords in Patient Medical Data')
plt.xlabel('Keyword')
plt.ylabel('Frequency')
plt.show()


# In[16]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt

# Define the list of irrelevant words to remove
irrelevant_words = ['pt', 'surgeon', 'would', 'b']

# Convert the DATE_RECEIVED column to datetime format
file1_df['DATE_RECEIVED'] = pd.to_datetime(file1_df['DATE_RECEIVED'], format='%Y-%m-%d')

# Tokenize the FOI_TEXT column
stop_words = set(stopwords.words('english'))
file1_df['tokens'] = file1_df['FOI_TEXT'].apply(lambda x: [word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words and word.lower() not in irrelevant_words])

# Extract keywords from the tokens column
keywords = Counter()
for token_list in file1_df['tokens']:
    keywords.update(token_list)
    
# Select the top 10 keywords and create a bar chart
top_keywords = dict(keywords.most_common(10))
plt.bar(top_keywords.keys(), top_keywords.values())
plt.title('Top 10 Keywords in Patient Medical Data')
plt.xlabel('Keyword')
plt.ylabel('Frequency')
plt.show()


# In[17]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt

# Define the list of irrelevant words to remove
irrelevant_words = ['pt', 'surgeon', 'would', 'b']

# Convert the DATE_RECEIVED column to datetime format
file1_df['DATE_RECEIVED'] = pd.to_datetime(file1_df['DATE_RECEIVED'], format='%Y-%m-%d')

# Tokenize the FOI_TEXT column
stop_words = set(stopwords.words('english'))
file1_df['tokens'] = file1_df['FOI_TEXT'].apply(lambda x: [word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words and word.lower() not in irrelevant_words])

# Extract keywords from the tokens column
keywords = Counter()
for token_list in file1_df['tokens']:
    keywords.update(token_list)
    
# Select the top 20 keywords and create a bar chart
top_keywords = dict(keywords.most_common(20))
plt.barh(list(top_keywords.keys()), list(top_keywords.values()))
plt.title('Top 20 Keywords in Patient Medical Data')
plt.xlabel('Frequency')
plt.ylabel('Keyword')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# In[18]:


import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ne_chunk
from nltk.tree import Tree
import matplotlib.pyplot as plt

# Define the list of irrelevant words to remove
irrelevant_words = ['pt', 'surgeon', 'would', 'b']

# Convert the DATE_RECEIVED column to datetime format
file1_df['DATE_RECEIVED'] = pd.to_datetime(file1_df['DATE_RECEIVED'], format='%Y-%m-%d')

# Tokenize the FOI_TEXT column
stop_words = set(stopwords.words('english'))
file1_df['tokens'] = file1_df['FOI_TEXT'].apply(lambda x: [word.lower() for word in word_tokenize(x) if word.isalpha() and word.lower() not in stop_words and word.lower() not in irrelevant_words])

# Perform entity recognition on the tokens column
def get_entities(text):
    chunked = ne_chunk(nltk.pos_tag(text))
    entities = []
    for node in chunked:
        if type(node) == Tree and node.label() in ['ORGANIZATION', 'PERSON']:
            entities.append(' '.join([word for word, pos in node.leaves()]))
    return entities

file1_df['entities'] = file1_df['tokens'].apply(get_entities)

# Flatten the list of entities and count the frequency of each entity
entity_counts = pd.Series([entity for sublist in file1_df['entities'] for entity in sublist]).value_counts()

# Select the top 10 entities and create a bar chart
top_entities = entity_counts.head(10)
plt.barh(top_entities.index, top_entities.values)
plt.title('Top 10 Entities in Patient Medical Data')
plt.xlabel('Frequency')
plt.ylabel('Entity')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# In[ ]:




