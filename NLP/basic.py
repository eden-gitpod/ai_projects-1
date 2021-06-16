#!/usr/bin/env python
# coding: utf-8

# In[1]:

import nltk

nltk.download('gutenberg')
nltk.download('genesis')
nltk.download('inaugural')
nltk.download('nps_chat')
nltk.download('webtext')
nltk.download('treebank')
nltk.download('punkt')

from nltk.book import *

# In[2]:


text1


# In[3]:


sents()


# In[4]:


sent1


# In[11]:


print(text7, len(text7))


# In[9]:


print(sent7, len(sent7))


# In[12]:


list(set(text7))[:10]


# In[13]:


# Frequency of words
dist = FreqDist(text7)
len(dist)


# In[18]:


vocab1 = list(dist.keys())
vocab1[:10]


# In[19]:


dist['Vinken']


# In[25]:


freqwords = [w for w in vocab1 if len(w) > 5 and dist[w] > 100]
freqwords


# In[28]:


# different forms of the same "word"
input1 = 'List listed lists listing listings'
words1 = input1.lower().split(' ')
words1


# In[29]:


porter = nltk.PorterStemmer()
[porter.stem(t) for t in words1]


# In[34]:


# tokenization
text11 = "Children shouldn't drink a sugary drink before bed."
text11.split(' ')


# In[35]:


nltk.word_tokenize(text11)


# In[36]:


# sentence splitting
text12 = 'This is the first sentence. A gallon of milk in the U.S. costs $2.99. Is this the third sentence? Yes, it is!'
sentences = nltk.sent_tokenize(text12)
len(sentences)


# In[37]:


print(sentences)

