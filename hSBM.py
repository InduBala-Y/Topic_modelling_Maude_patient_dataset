#!/usr/bin/env python
# coding: utf-8

# In[ ]:





import numpy as np
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import networkx as nx
import community  # Louvain algorithm for community detection
import pymc3 as pm

# Preprocessing
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')

def preprocess_text(text):
    # Tokenize, remove stop words, and lowercase the text
    tokens = nltk.word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha() and token.lower() not in stopwords]
    return ' '.join(tokens)

file1_df['FOI_TEXT'] = file1_df['FOI_TEXT'].apply(preprocess_text)

# Create document-term matrix
vectorizer = CountVectorizer()
dtm = vectorizer.fit_transform(file1_df['FOI_TEXT'])



# Model Setup
num_topics = 5  # Specify the desired number of topics
num_terms = len(vectorizer.get_feature_names())
num_docs = dtm.shape[0]

# Initialize topic assignments randomly
np.random.seed(42)
z = np.random.randint(low=0, high=num_topics, size=(num_docs, num_terms))

# Gibbs Sampling
alpha = np.ones(num_topics)  # Prior for document-topic distribution
beta = np.ones((num_topics, num_terms))  # Prior for topic-term distribution

num_iterations = 1000  # Number of Gibbs sampling iterations
for iteration in range(num_iterations):
    # For each document in the corpus
    for doc_idx in range(num_docs):
        doc_term_indices = dtm[doc_idx].nonzero()[1]
        
        # For each word in the document
        for term_idx in doc_term_indices:
            # Remove the word's current topic assignment
            topic = z[doc_idx, term_idx]
            z[doc_idx, term_idx] = -1
            
            # Calculate the probabilities of assigning the word to each topic
            topic_counts = np.sum(z[doc_idx] == np.arange(num_topics)[:, None], axis=1)
            term_counts = np.sum(z == topic, axis=1)
            p_topic = (topic_counts + alpha) / (np.sum(topic_counts) + num_topics * alpha.sum())
            p_term = (term_counts + beta[:, term_idx]) / (topic_counts + num_terms * beta.sum(axis=1))
            p = p_topic * p_term
            p /= np.sum(p)
            
            # Sample a new topic assignment for the word
            z[doc_idx, term_idx] = np.random.choice(np.arange(num_topics), p=p)
    
    # Update document-topic and topic-term distributions
    theta = np.zeros((num_docs, num_topics))
    phi = np.zeros((num_topics, num_terms))
    for doc_idx in range(num_docs):
        for topic_idx in range(num_topics):
            theta[doc_idx, topic_idx] = np.sum(z[doc_idx] == topic_idx) / dtm[doc_idx].sum()
    
    for topic_idx in range(num_topics):
        for term_idx in range(num_terms):
            phi[topic_idx, term_idx] = np.sum(z[:, term_idx] == topic_idx) / np.sum(z == topic_idx)

            
# Create a network based on document similarity
similarity_matrix = dtm.dot(dtm.T)
network = nx.from_scipy_sparse_matrix(similarity_matrix)

# Perform community detection using Louvain algorithm
partition = community.best_partition(network)

# Add community information to the DataFrame
file1_df['Community'] = pd.Series(partition)
    
# Estimate the community structure based on the inferred document-topic distributions

# Analyzing Results
top_terms_per_topic = 30
for topic_idx in range(num_topics):
    top_term_indices = np.argsort(phi[topic_idx])[::-1][:top_terms_per_topic]


# In[ ]:


# Optimal number of Topics

# Compute Coherence Score
coherence_model = CoherenceModel(
    topics=[vectorizer.get_feature_names()[term_idx] for term_idx in range(num_terms)],
    texts=file1_df['FOI_TEXT'].apply(nltk.word_tokenize).


# In[ ]:




