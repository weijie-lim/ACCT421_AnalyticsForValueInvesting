"""
FILE:  snp-performance-metrics(revised).py
THIS PROGRAM CALCULATES SHARPE RATIO, SORTINO RATIO, MAXIMUM DRAWDOWN
AND MAXIMUM DRAWDOWN DURATION FOR THE MONTHLY S&P 500 INDEX
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Set path of my sub-directory
from pathlib import Path
myfolder = Path(r'C:\Users\User\Documents')

# 2. Read stock price data of S&P500
snp = pd.read_csv(myfolder / 'S&P500.csv', parse_dates=['Date'])

# 3. Compute monthly stock returns from price data
snp['mret'] = snp.sort_values(by='Date')['Adj Close'].pct_change()
snp['mret1'] = 1 + snp['mret']
#    Rename 'Adj Close' to 'price' and retain relevant variables
snp = snp.rename(columns = {'Adj Close': 'price'})[['Date', 'price', 'mret', 'mret1']]
print('Descriptive statistics of monthly S&P500 returns \n', snp['mret'].describe(), end='\n'*5)

# 4. Compute annualized Sharpe ratio with monthly returns using numpy library functions
sharpe = (np.mean(snp['mret']) * 12) / (np.std(snp['mret'], ddof=1) * np.sqrt(12))
print('Sharpe ratio (using numpy) = ', sharpe.round(4), end='\n'*5)

# 4a. Compute annualized Sharpe ratio (alternative method using pandas library functions,  
#     since snp is a pandas dataframe)
sharpe2 = (snp['mret'].mean() * 12) / (snp['mret'].std() * np.sqrt(12))
print('Sharpe ratio (using pandas) = ', sharpe2.round(4), end='\n'*5)


#4b. Sortino ratio 
#2c.Annual Sortino ratio (assume zero risk-free rate as benchmark).
length = len(snp.index)/12

snpmean = snp['mret'].mean()
print(snpmean)
mask = (snp['mret'] < snpmean)
snp = snp.loc[mask]

snp['Sq Diff from Mean'] = (snp['mret'] - snpmean)**2
sofsd = snp['Sq Diff from Mean'].sum()
print('The sum of the squared difference from the mean is', sofsd.round(4),end='\n'*3)
semisd = np.sqrt(sofsd*(1/((length*12)-1)))
print('Semi-Standard Deviation is',semisd.round(4),'.',end='\n'*3)


Sortino = (snpmean*12)/semisd

print('Sortino Ratio is',Sortino.round(4),'.',end='\n'*3)

hwm = np.zeros(len(snp))#Np.zero using a function from the nup pie lib 
drawdown = np.zeros(len(snp))#to initialise from the nplibrary
duration = 0


"""
# 5. Compute maximum drawdown (in %) and drawdown duration (in months)
"""
cumret1 = np.cumprod(snp['mret1'])
hwm1 = cumret1.cummax()
drawdown1 = ((hwm1 - cumret1) / hwm1) * 100
print('Maximum Drawdown = ', drawdown1.round(2).max(), '%', end='\n'*5)

# 6. Plot S&P 500 Index
fig, axes = plt.subplots(figsize=(8,6))
axes.plot(snp.Date, snp.price)
axes.set(xlabel = 'Date', ylabel = 'Index', title = 'S&P 500 Index')
plt.show()