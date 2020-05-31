import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import mean_squared_error
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from numpy import inf
from math import exp, gamma
from datetime import timedelta
from sklearn.metrics import r2_score
import matplotlib.patheffects as PathEffects
from scipy.special import softmax
import warnings
import os
import math
from scipy.stats import pearsonr, spearmanr

warnings.simplefilter("ignore")

plt.style.use(['science'])
plt.rcParams["text.usetex"] = True

indicators = ['Population ages 65 and above (% of total population)', \
	'Population ages 15-64 (% of total population)',\
	'Population ages 0-14 (% of total population)', \
	'People with basic handwashing facilities including soap and water (% of population)',\
	'Average Yearly Temperature (C)',\
	'O', 'B', 'B1','B2', 'B4', 'A3', 'A6', 'A7', 'A1a', 'A2', 'A2a',\
	'Trade with China Exports + Import US$ billion 2018',\
	'Air transport, passenger carried 2018 (million) WB',\
	'Stringency Score Avg per day after 100 patients reported']


params = ['peaks diff', 'total cases', 'total deaths', 'cases/pop', 'deaths/pop', 'mortality', 'k new', 'a new', 'b new', 'g new', 'k dead', 'a dead', 'b dead', 'g dead']
df = pd.read_excel('correlation.xlsx', sheet_name='Raw Data (deaths)')
df.replace([np.inf, -np.inf, np.nan, ''], 0, inplace=True)

corrfunc = pearsonr

correlationdata = []; pdata  = []
for i in indicators:
	result = [corrfunc(df[p],df[i]) for p in params]
	correlationdata.append([i] + [res[0] for res in result])
	pdata.append([i] + [res[1] for res in result])

df2 = pd.DataFrame(correlationdata,columns=['Indicator']+params)
df2p = pd.DataFrame(pdata, columns=['Indicator']+params)

with pd.ExcelWriter('correlation.xlsx') as writer:  
    df.to_excel(writer, sheet_name='Raw Data')
    df2.to_excel(writer, sheet_name='Correlation Data')
    df2p.to_excel(writer, sheet_name='Significance (p value)')