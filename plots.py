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

df = pd.read_csv('owid-covid-data.csv')
df['date'] = pd.to_datetime(df.date)

countries = list(pd.unique(df['location']))

def weib(x, k, a, b, g):
	return k * g * b * (a ** b) * np.exp(-1 * g * ((a / x)  ** b)) / (x ** (b + 1))

def getInfoCountry(df2, isdead):
	df2['Delta'] = (df2.date - min(df2.date)).dt.days
	startDate = min(df2.date)
	totalLength = max(df2.Delta)
	confirmed = []; new = []
	for day in range(totalLength):
		newc = max(0, int(sum(df2.new_cases[df2.Delta == day] if not isdead else df2.new_deaths[df2.Delta == day])))
		new.append(newc)
		confirmed.append(new[-1] + (confirmed[-1] if len(confirmed) > 1 else 0))
	return startDate, totalLength, confirmed, new

def totalExpected(func, popt, data):
	total = 0; day = 1
	while True:
		today = func(day, *popt) if day >= len(data) else data[day]
		total += today
		day += 1
		if day > data.index(max(data)) and today <= 1: break
	return day, total

def calcWhen(func, popt, match, data):
	total = 0; day = 1
	while True:
		today = func(day, *popt) if day >= len(data) else data[day]
		total += today
		day += 1
		if total >= match or (today == 0 and day > data.index(max(data))): break
	return day

def iterativeCurveFit(func, x, y, start):
	outliersweight = None
	for i in range(10):
		popt, pcov = curve_fit(func, x, y, start, sigma=outliersweight, maxfev=100000)
		pred = np.array([func(px, *popt) for px in x])
		old = outliersweight
		outliersweight = np.abs(pred - y)
		outliersweight = 1 - np.tanh(outliersweight)
		outliersweight = outliersweight / np.max(outliersweight)
		outliersweight = softmax(1 - outliersweight)
		if i > 1 and sum(abs(old - outliersweight)) < 0.001: break
	return popt, pcov

def seriesIterativeCurveFit(func, xIn, yIn, start):
	res = []
	for ignore in range(15, 0, -1):
		x = xIn[:-1*ignore]; y = yIn[:-1*ignore]
		outliersweight = None
		for i in range(10):
			popt, pcov = curve_fit(func, x, y, start, sigma=outliersweight, absolute_sigma=True, maxfev=100000)
			pred = np.array([func(px, *popt) for px in x])
			old = outliersweight
			outliersweight = np.abs(pred - y)
			outliersweight = 1 - np.tanh(outliersweight)
			outliersweight = outliersweight / np.max(outliersweight)
			outliersweight = softmax(1 - outliersweight)
			if i > 1 and sum(abs(old - outliersweight)) < 0.001: break
		pred = [func(px, *popt) for px in xIn]
		res.append((mean_absolute_percentage_error(yIn, pred), popt, pcov, ignore))
	# for i in res: print(i)
	val = res[res.index(min(res))]
	return val[1], val[2]

def getMaxCases(y, data):
	m = 0; dday = 0
	for day,cases in enumerate(y):
		if day < len(data):
			if data[day] > m:
				m = data[day]; dday = day
		else:
			if cases > m:
				m = cases; dday = day
	return m, dday

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / (np.array(y_true)+1))) * 100

def getcummulative(l):
	res = []; s = 0
	for i in l:
		s += i; res.append(s)
	return res

dead = False
finaldata = []
dfPlot = pd.DataFrame()
training_data = -1
interactive = ['India', 'World', 'United States', 'United Kingdom', 'China', 'Spain', 'Italy', 'France', 'Germany', 'Russia']
for country in interactive:
	try:
		dead = False
		print("--", country)
		df2 = df[df['location'] == country]
		res = getInfoCountry(df2, False)
		data = res[-1]
		if sum(data) < (2000 if not dead else 100) and not data in ['Brazil', 'Iran', 'Israel', 'Oman']:
			print('skip', country,)
			continue
		days = res[1]
		start = res[0]

		func = [(weib, [0, 20, 100]), (weib, [60000, 14, 4, 500]), (weib, [7000, 0.5, 0.001, 100])]

		whichFunc = 1
		times = 2; skip = 30
		datacopy = deepcopy(data[1:training_data]); x = list(range(len(data)))
		if country == 'China': datacopy[datacopy == 15141] = 4000
		popt, pcov = seriesIterativeCurveFit(func[whichFunc][0], x[1:training_data], datacopy, func[whichFunc][1])
		finalday, finalexp = totalExpected(func[whichFunc][0], popt, data)
		when97 = calcWhen(func[whichFunc][0], popt, 0.97 * finalexp, data)

		when97 = 1000 if when97 > 1000 else when97
		xlim = max(len(data)*times, when97+10)
		pred = [func[whichFunc][0](px, *popt) for px in list(range(xlim))[1:]]

		y = [func[whichFunc][0](px, *popt) for px in list(range(xlim))[1:]]
		maxcases, maxday = getMaxCases(y, data)
		newpredsave = deepcopy(y)
		newsave = deepcopy(data)
		cumpredsave = getcummulative(y)

		dead = True
		res = getInfoCountry(df2, True)
		data = res[-1]
		deadsave = deepcopy(data)

		xlim2 = max(len(data)*times, when97+10)
		xlim = max(xlim, xlim2)

		datacopy = np.absolute(np.array(deepcopy(data[1:training_data])))
		poptold = popt
		finalexpold = finalexp
		popt, pcov = seriesIterativeCurveFit(func[whichFunc][0], x[1:training_data], datacopy, [2000, 54, 4, 500])
		y = [func[1][0](px, *popt) for px in x[1:]]
		pred = [func[whichFunc][0](px, *popt) for px in list(range(xlim2))[1:]]
		deadpredsave = deepcopy(pred)

		newdf = pd.DataFrame.from_dict({country+'-pred': newpredsave, \
			country+'-dates':[(start+timedelta(days=i)).strftime("%d %b %Y") for i in list(range(1,xlim))], \
			country+'-true': newsave,\
			country+'-cum': cumpredsave,\
			country+'-predd': deadpredsave,\
			country+'-trued': deadsave}, orient='index').T
		dfPlot[country+'-pred'] = pd.Series(newdf[country+'-pred'])
		dfPlot[country+'-dates'] = pd.Series(newdf[country+'-dates'])
		dfPlot[country+'-true'] = pd.Series(newdf[country+'-true'])
		dfPlot[country+'-cum'] = pd.Series(newdf[country+'-cum'])
		dfPlot[country+'-predd'] = pd.Series(newdf[country+'-predd'])
		dfPlot[country+'-trued'] = pd.Series(newdf[country+'-trued'])
	except Exception as e:
		print(str(e))
		# raise(e)
		pass


dfPlot.to_excel('plot.xlsx')
