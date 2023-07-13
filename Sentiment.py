#!/usr/bin/env python
# coding: utf-8

# In[75]:


import numpy as np
import pandas as pd
import statsmodels.api as sma
import scipy.stats as scs
from scipy.stats.mstats import winsorize
from matplotlib import pyplot
from numpy import mean
from numpy import std
from numpy.random import seed
from numpy import cov
import matplotlib.pyplot as plt




# 1. Set path of my sub-directory
from pathlib import Path
myfolder = Path('C:/Users/User/OneDrive - Singapore Management University/Analytics for Value investing')

# 2. Read stock price data of S&P500
df = pd.read_csv(myfolder / 'Sentiment Analysis.csv')
sentiment_list = []
counter=0

from textblob import TextBlob

textblob_score = []

for sentence in df["headline_text"]:
    sentiment = TextBlob(str(sentence)).sentiment
    textblob_score.append(sentiment.polarity)
df['textblob_score'] = textblob_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()  

vader_score = []

for sentence in df["headline_text"]:
    sentiment = analyzer.polarity_scores(sentence)
    vader_score.append(sentiment['compound'])
df['vader_score'] = vader_score
df['averagescore'] =(df['textblob_score'] + df['vader_score'])/2
df1 = df[['averagescore', 'returns']]
df1['returns']=df1['returns'].str.replace('%', '')
pyplot.scatter(df1['returns'], df1['averagescore'])
plt.title('Sentiment Scatter Plot')
plt.ylabel('Sentiment Scores')
plt.xlabel('Returns')
pyplot.show()
df1['returns']=df1['returns'].astype(np.float)
print('Correlation coefficients of variables\n',  
          df1[['returns','averagescore']].corr(), end='\n'*4) #good to compute correlation for all variable


# In[ ]:




