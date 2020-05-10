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
import torch.nn.functional as F
from numpy import inf
from math import exp
from datetime import timedelta
from sklearn.metrics import r2_score
from scipy.special import softmax
import warnings
import os

warnings.simplefilter("ignore")

plt.style.use(['science'])
plt.rcParams["text.usetex"] = True

df = pd.read_excel('misc.xlsx')
countries = df['Country']
data = df['Mortality rate']

plt.figure(figsize=(8,3))
bars = plt.bar(list(range(len(data))), data, width=1, edgecolor='black', linewidth=0.01, alpha=1, label='Mortality Rate %')
bars[0].set_color('r')
bars[0].set_edgecolor('k')
plt.xticks([i+0.5 for i in list(range(len(data)))], countries, rotation=90, ha='right')

plt.xlabel('Countries')
plt.ylabel('Predicted Mortality Rate \%')

plt.savefig('mortality.pdf')