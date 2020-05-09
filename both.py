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
from math import exp, gamma
from datetime import timedelta
from sklearn.metrics import r2_score
import matplotlib.patheffects as PathEffects
from scipy.special import softmax
import warnings
import os

warnings.simplefilter("ignore")

dead = True

plt.style.use(['science'])
plt.rcParams["text.usetex"] = True

df = pd.read_csv('owid-covid-data1.csv')
dfOld = pd.read_csv('owid-covid-data.csv')
df['Date'] = pd.to_datetime(df.Date)
dfOld['Date'] = pd.to_datetime(dfOld.Date)

countries = list(pd.unique(df['Country']))

def gauss(x, mu, sigma, scale):
    return scale * np.exp(-1 * ((x - mu) ** 2) / (2 * (sigma ** 2) )) 

def weib(x, k, a, b, g):
	return k * g * b * (a ** b) * np.exp(-1 * g * ((a / x)  ** b)) / (x ** (b + 1))

def beta(x, k, a, b, p, q):
	return k * gamma(p + q) * ((x - a)** (p-1)) * (b-x)**(q-1) / (gamma(p) * gamma(q) * (b-a)**(p+q-1))

def ft(x, k, e, d, o):
	return k * np.exp(-1 * (1 + e * (x-o)) ** (-1 / (e + d)))

def getInfos(df2):
	df2['Delta'] = (df2.Date - min(df2.Date)).dt.days
	startDate = min(df2.Date)
	totalLength = max(df2.Delta)
	confirmed = []; new = []
	for day in range(totalLength):
		if not df2.Confirmed[df2.Delta == day].empty:
			lastconfirmed = int(sum(df2.Confirmed[df2.Delta == day]))  
		else:
			confirmed[-1] if confirmed != [] else 0
		confirmed.append(lastconfirmed)
		new.append(confirmed[-1] - (confirmed[-2] if len(confirmed) > 1 else 0))
	return startDate, totalLength, confirmed, new

def getSars():
	df2 = pd.read_csv('sars_2003_complete_dataset_clean.csv')
	# df2 = df2[df2['Country'] == 'Vietnam']
	df2['Date'] = pd.to_datetime(df2.Date, format="%Y-%m-%d")
	df2['Delta'] = (df2.Date - min(df2.Date)).dt.days
	startDate = min(df2.Date)
	totalLength = max(df2.Delta)
	confirmed = []; new = []; conf = 0
	for day in range(totalLength):
		conf = max(conf, int(sum(df2.confirmed[df2.Delta == day]))  )
		confirmed.append(conf)
		new.append(confirmed[-1] - (confirmed[-2] if len(confirmed) > 1 else 0))
	print(new)
	return [[startDate, totalLength, confirmed, new], df2]

def getInfoCountry(df2, isdead):
	df2['Delta'] = (df2.Date - min(df2.Date)).dt.days
	startDate = min(df2.Date)
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
		if day > len(data) and today <= 1: break
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
	for i in range(100):
		popt, pcov = curve_fit(func, x, y, start, sigma=outliersweight, maxfev=100000)
		pred = np.array([func(px, *popt) for px in x])
		outliersweight = np.abs(pred - y)
		outliersweight = 1 - np.tanh(outliersweight)
		outliersweight = outliersweight / np.max(outliersweight)
		outliersweight = softmax(1 - outliersweight)
	return popt, pcov

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

insufficient = ['Central African Republic', 'Cambodia', 'Sudan', 'Ecuador', 'Chile', 'Peru'] 
finaldata = []
for country in countries:
	if country in insufficient:
		continue
	# if os.path.exists('graphs/'+'both'+'/'+country.replace(" ", "_")+'.pdf'): continue
	try:
		dead = False
		print("--", country)
		df2 = df[df['Country'] == country] if country not in ['Russia'] else dfOld[dfOld['Country'] == country]
		res = getInfoCountry(df2, False)
		data = res[-1]
		if sum(data) < (2000 if not dead else 100) and not data in ['Brazil', 'Iran', 'Israel', 'Oman']:
			print('skip', country,)
			continue
		days = res[1]
		start = res[0]

		func = [(gauss, [0, 20, 100]), (weib, [30000, 14, 4, 500]), (ft, [3000, 0.5, 0.001, 100])]

		whichFunc = 0
		times = 2
		plt.figure(figsize=(6,3))
		x = list(range(len(data)))
		datacopy = np.array(deepcopy(data[1:]))
		if country == 'China': datacopy[datacopy == 15141] = 4000
		poptg, pcovg = curve_fit(func[whichFunc][0], x[1:], datacopy, func[whichFunc][1], maxfev=100000)
		whichFunc = 1
		popt, pcov = iterativeCurveFit(func[whichFunc][0], x[1:], datacopy, func[whichFunc][1])
		finalday, finalexp = totalExpected(func[whichFunc][0], popt, data)
		when97 = calcWhen(func[whichFunc][0], popt, 0.97 * finalexp, data)

		xlim = max(len(data)*times, when97+10)
		pred = [func[whichFunc][0](px, *popt) for px in list(range(xlim))[1:]]

		plt.plot(list(range(xlim))[1:], pred, color='red', label='Robust Weibull Prediction (new)')
		print("MSE ", "{:e}".format(mean_squared_error(data[1:], [func[whichFunc][0](px, *popt) for px in x[1:]])))
		print("R2 ", "{:e}".format(r2_score(data[1:], [func[whichFunc][0](px, *popt) for px in x[1:]])))
		# print("97 day", start + timedelta(days=when97))
		# print("final day", start + timedelta(days=finalday))
		# print("total cases", finalexp)
		_ = plt.bar(x, data, width=1, edgecolor='black', linewidth=0.01, alpha=0.2, label='Actual Data (new)')
		dt = list(df2.Date)
		skip = 30

		# Metrics
		y = [func[1][0](px, *popt) for px in x[1:]]
		mse = "{:e}".format(mean_squared_error(data[1:], y))
		mape = "{:e}".format(mean_absolute_percentage_error(data[1:], y))
		mseg = "{:e}".format(mean_squared_error(data[1:], [func[0][0](px, *poptg) for px in x[1:]]))
		mapeg = "{:e}".format(mean_absolute_percentage_error(data[1:], [func[0][0](px, *poptg) for px in x[1:]]))
		r2 = "{:e}".format(r2_score(data[1:], y))
		r2g = "{:e}".format(r2_score(data[1:], [func[0][0](px, *poptg) for px in x[1:]]))
		y = [func[whichFunc][0](px, *popt) for px in list(range(xlim))[1:]]
		maxcases, maxday = getMaxCases(y, data)
		print(mape, mapeg)
		style = dict( arrowstyle = "-" ,  connectionstyle = "angle", ls =  'dashed')
		plt.ylabel("Number of cases")
		plt.xlabel("Date")
		plt.tight_layout()
		folder = 'both'
		plt.legend(loc='best')
		plt.title(country)

		dead = True
		print("--", country)
		df2 = df[df['Country'] == country]
		res = getInfoCountry(df2, True)
		data = res[-1]

		xlim2 = max(len(data)*times, when97+10)
		xlim = max(xlim, xlim2)

		plt.xticks(list(range(0,xlim,30)), [(start+timedelta(days=i)).strftime("%d %b %y") for i in range(0,xlim,skip)], rotation=45, ha='right')
		plt.twinx()


		datacopy = np.array(deepcopy(data[1:]))
		poptold = popt
		popt, pcov = iterativeCurveFit(func[whichFunc][0], x[1:], datacopy, func[whichFunc][1])
		finalday, finalexp = totalExpected(func[whichFunc][0], popt, data)
		when97 = calcWhen(func[whichFunc][0], popt, 0.97 * finalexp, data)
		pred = [func[whichFunc][0](px, *popt) for px in list(range(xlim2))[1:]]
		maxcases2, maxday2 = getMaxCases(pred, data)

		plt.plot(list(range(xlim2))[1:], pred, color='purple', label='Robust Weibull Prediction (dead)')
		_ = plt.bar(x, data, width=1, color='green', edgecolor='black', linewidth=0.01, alpha=0.2, label='Actual Data (dead)')
		plt.legend(loc=7)
		plt.ylabel("Number of deaths")
		xlim = max(xlim, xlim2)

		plt.savefig('graphs/'+'both'+'/'+country.replace(" ", "_")+'.pdf')
		values = [country, (start+timedelta(days=maxday)).strftime("%d %b %Y"), (start+timedelta(days=maxday2)).strftime("%d %b %Y"), abs(maxday-maxday2)]
		values.extend(poptold)
		values.extend(popt)
		finaldata.append(values)
		print(country)
	except Exception as e:
		print(str(e))
		# raise(e)
		pass

df = pd.DataFrame(finaldata,columns=['Country','Max day (cases)','Max day (deaths)', 'Difference (days)', 'k (new)', 'a (new)', 'b (new)', 'g (new)', 'k (dead)', 'a (dead)', 'b (dead)', 'g (dead)'])
df.to_excel('params.xlsx')
