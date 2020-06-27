import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import mean_squared_error
import numpy as np
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
from tqdm import tqdm
import math
from scipy.stats import pearsonr, spearmanr

warnings.simplefilter("ignore")

plt.style.use(['science'])
plt.rcParams["text.usetex"] = True

df = pd.read_csv('state_wise_daily.csv')
df['date'] = pd.to_datetime(df.Date)

def gauss(x, mu, sigma, scale):
    return scale * np.exp(-1 * ((x - mu) ** 2) / (2 * (sigma ** 2) )) 

def weib(x, k, a, b, g):
	return k * g * b * (a ** b) * np.exp(-1 * g * ((a / x)  ** b)) / (x ** (b + 1))

def beta(x, k, a, b, p, q):
	return k * gamma(p + q) * ((x - a)** (p-1)) * (b-x)**(q-1) / (gamma(p) * gamma(q) * (b-a)**(p+q-1))

def ft(x, k, e, d, o):
	return k * np.exp(-1 * (1 + e * (x-o)) ** (-1 / (e + d)))

def getInfoCountry(df2, isdead):
	df2['Delta'] = (df2.date - min(df2.date)).dt.days
	startDate = min(df2.date)
	totalLength = max(df2.Delta)
	confirmed = []; new = []
	cols = list(df2.columns)[2:-2]
	for day in range(int(totalLength)):
		newc = int(df2[cols][df2.Delta == day][df.Status == ('Confirmed' if not isdead else 'Deceased')].sum(axis=1))
		new.append(newc)
		confirmed.append(new[-1] + (confirmed[-1] if len(confirmed) > 1 else 0))
	return startDate, totalLength, confirmed, new

def totalExpected(func, popt, data):
	total = 0; day = 1
	while True:
		today = func(day, *popt) if day >= len(data) else data[day]
		total += today
		day += 1
		if (day > len(data) and today <= 1) or day > 1000: break
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
		outliersweight = outliersweight - np.tanh(outliersweight)
		outliersweight = outliersweight / np.max(outliersweight)
		outliersweight = softmax(1 - outliersweight)
		if i > 1 and sum(abs(old - outliersweight)) < 0.001: break
	return popt, pcov

def seriesIterativeCurveFit(func, xIn, yIn, start):
	res = []
	for ignore in tqdm(list(range(10, 0, -1)), ncols=80):
		x = xIn[:-1*ignore]; y = yIn[:-1*ignore]
		outliersweight = None
		for i in range(10):
			popt, pcov = curve_fit(func, x, y, start, sigma=outliersweight, absolute_sigma=True, maxfev=1000000)
			pred = np.array([func(px, *popt) for px in x])
			old = outliersweight
			outliersweight = np.abs(pred - y)
			outliersweight = outliersweight - np.tanh(outliersweight)
			outliersweight = outliersweight / np.max(outliersweight)
			outliersweight = softmax(1 - outliersweight)
			if i > 1 and sum(abs(old - outliersweight)) < 0.001: break
		pred = [func(px, *popt) for px in xIn]
		res.append((mean_absolute_percentage_error(yIn, pred), popt, pcov, ignore))
	# for i in res: print(i)
	errors = [i[0] for i in res]
	val = res[errors.index(min(errors))]
	return val[1], val[2]

def getMaxCases(y, data):
	m = 0; dday = 0
	for day,cases in enumerate(y):
		# if day < len(data):
		# 	if data[day] > m:
		# 		m = data[day]; dday = day
		# else:
			if cases > m:
				m = cases; dday = day
	return m, dday

def mean_absolute_percentage_error(y_true, y_pred): 
    return np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / (np.array(y_true)+1))) * 100

insufficient = ['Central African Republic', 'Cambodia', 'Sudan', 'Ecuador', 'Chile', 'Colombia', 'Peru'] 
finaldata = []; gooddataNew = []; gooddataDead = []
ignore = -1
training_data = -15
for country in ['India']:
	if country in insufficient:
		continue
	try:
		dead = False
		print("--", country)
		df2 = df
		res = getInfoCountry(df2, False)
		data = res[-1]
		days = res[1]
		start = res[0]

		func = [(gauss, [0, 20, 100]), (weib, [160000, 14, 4, 500]), (ft, [7000, 0.5, 0.001, 100])]

		whichFunc = 0
		times = 2; skip = 30
		plt.figure(figsize=(6,3))
		x = list(range(len(data)))
		datacopy = np.absolute(np.array(deepcopy(data[1:training_data])))
		if country == 'China': datacopy[datacopy == 15141] = 4000
		# poptg, pcovg = curve_fit(func[whichFunc][0], x[1:training_data], datacopy, func[whichFunc][1], maxfev=100000)
		whichFunc = 1
		popt, pcov = seriesIterativeCurveFit(func[whichFunc][0], x[1:training_data], datacopy, func[whichFunc][1])
		finalday, finalexp = totalExpected(func[whichFunc][0], popt, data)
		when97 = calcWhen(func[whichFunc][0], popt, 0.97 * finalexp, data)

		when97 = 1000 if when97 > 1000 else when97
		xlim = max(len(data)*times, 180)
		pred = [func[whichFunc][0](px, *popt) for px in list(range(xlim))[1:]]

		plt.plot(list(range(xlim))[1:], pred, color='red', label='Robust Weibull Prediction (new)')
		_ = plt.bar(x, data, width=1, edgecolor='black', linewidth=0.01, alpha=0.2, label='Actual Data (new)')
		plt.ylabel("Number of cases"); plt.xlabel("Date"); plt.tight_layout(); 
		plt.legend(loc='best');	plt.title(country)

		y = [func[1][0](px, *popt) for px in x[1:]]
		r2 = r2_score(data[1:], y)
		mape = mean_absolute_percentage_error(data[1:], y)

		print("MSE ", "{:e}".format(mean_squared_error(data[1:], y)))
		print("R2 ", r2)
		print("97 day", (start + timedelta(days=when97)).strftime("%d %b %y"))
		print("MAPE", mape)
		mape_error_new = mean_absolute_percentage_error(data[training_data:], y[training_data:])
		r2_error_new = r2_score(data[training_data:], y[training_data:])

		# Metrics
		y = [func[whichFunc][0](px, *popt) for px in list(range(xlim))[1:]]
		maxcases, maxday = getMaxCases(y, data)

		dead = True
		res = getInfoCountry(df2, True)
		data = res[-1]

		xlim2 = max(len(data)*times, 180)

		xlim = max(xlim, xlim2)
		plt.xticks(list(range(0,xlim,30)), [(start+timedelta(days=i)).strftime("%d %b %y") for i in range(0,xlim,skip)], rotation=45, ha='right')
		plt.twinx()

		datacopy = np.absolute(np.array(deepcopy(data[1:training_data])))
		poptold = popt
		finalexpold = finalexp
		popt, pcov = seriesIterativeCurveFit(func[whichFunc][0], x[1:training_data], datacopy, [160000, 14, 4, 500])
		y = [func[1][0](px, *popt) for px in x[1:]]
		r2Dead = r2_score(data[1:], y)
		mapeDead = mean_absolute_percentage_error(data[1:], y)
		mape_error_dead = mean_absolute_percentage_error(data[training_data:], y[training_data:])
		r2_error_dead = r2_score(data[training_data:], y[training_data:])
		finalday, finalexp = totalExpected(func[whichFunc][0], popt, data)
		pred = [func[whichFunc][0](px, *popt) for px in list(range(xlim2))[1:]]
		maxcases2, maxday2 = getMaxCases(pred, data)
		plt.plot(list(range(xlim2))[1:], pred, color='purple', label='Robust Weibull Prediction (dead)')
		_ = plt.bar(x, data, width=1, color='green', edgecolor='black', linewidth=0.01, alpha=0.2, label='Actual Data (dead)')
		plt.legend(loc=7)
		plt.ylabel("Number of deaths")

		plt.savefig('India_ICMR.pdf')

		population = 1379715223
		values = [country, mape_error_new, mape_error_dead, r2_error_new, r2_error_dead, r2, mape, r2Dead, mapeDead, maxday2-maxday, finalexpold, finalexp, finalexpold/population, finalexp/population, 100*finalexp/finalexpold]
		finaldata.append(values)
		if maxday2 - maxday >= -10 and mape <= 46: 
			gooddataNew.append(finaldata[-1])
		if maxday2 - maxday >= -10 and mapeDead <= 47: 
			gooddataDead.append(finaldata[-1])
		print("----", country)
	except Exception as e:
		print(str(e))
		raise(e)
		pass

params = ['peaks diff', 'total cases', 'total deaths', 'cases/pop', 'deaths/pop', 'mortality']
df = pd.DataFrame(finaldata,columns=['Country', 'Prediction MAPE (new)', 'Prediction MAPE (dead)', 'Prediction R2 (new)', 'Prediction R2 (dead)', 'R2', 'MAPE', 'R2 Deaths', 'MAPE Deaths']+params)
with pd.ExcelWriter('error.xlsx') as writer:  
    df.to_excel(writer, sheet_name='All errors')