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

df = pd.read_csv('owid-covid-data.csv')
df['Date'] = pd.to_datetime(df.Date)

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
	df2 = pd.read_csv('datasets/sars_2003_complete_dataset_clean.csv')
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

def getInfoCountry(df2):
	df2['Delta'] = (df2.Date - min(df2.Date)).dt.days
	startDate = min(df2.Date)
	totalLength = max(df2.Delta)
	confirmed = []; new = []
	for day in range(totalLength):
		newc = max(0, int(sum(df2.new_cases[df2.Delta == day] if not dead else df2.new_deaths[df2.Delta == day])))
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
		popt, pcov = curve_fit(func, x, y, start, sigma=outliersweight, maxfev=10000)
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

insufficient = ['Central African Republic', 'Cambodia', 'Sudan', 'Ecuador', 'Chile'] 
finaldata = []
for country in countries:
	if country in insufficient:
		continue
	# if os.path.exists('graphs/'+country+'.pdf'): continue
	try:
		print("--", country)
		df2 = df[df['Country'] == country]
		res = getInfoCountry(df2)
		# res, df2 = getSars()
		# country = 'SARS'
		data = res[-1]
		if sum(data) < (2000 if not dead else 100) and not data in ['Brazil', 'Peru', 'Iran', 'Israel', 'Oman']:
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
		poptg, pcovg = curve_fit(func[whichFunc][0], x[1:], datacopy, func[whichFunc][1], maxfev=10000)
		whichFunc = 1
		popt, pcov = iterativeCurveFit(func[whichFunc][0], x[1:], datacopy, func[whichFunc][1])
		finalday, finalexp = totalExpected(func[whichFunc][0], popt, data)
		when97 = calcWhen(func[whichFunc][0], popt, 0.97 * finalexp, data)
		# finaldayg, finalexpg = totalExpected(func[0][0], poptg, data)
		# when97g = calcWhen(func[0][0], poptg, 0.97 * finalexpg, data)

		xlim = max(len(data)*times, when97+10)
		pred = [func[whichFunc][0](px, *popt) for px in list(range(xlim))[1:]]

		plt.plot(list(range(xlim))[1:], pred, color='red', label='Robust Weibull Prediction')
		plt.plot(list(range(xlim))[1:], [func[0][0](px, *poptg) for px in list(range(xlim))[1:]], '--', color='green', label='Gaussian Prediction')
		print("MSE ", "{:e}".format(mean_squared_error(data[1:], [func[whichFunc][0](px, *popt) for px in x[1:]])))
		print("R2 ", "{:e}".format(r2_score(data[1:], [func[whichFunc][0](px, *popt) for px in x[1:]])))
		print("97 day", start + timedelta(days=when97))
		print("final day", start + timedelta(days=finalday))
		print("total cases", finalexp)
		_ = plt.bar(x, data, width=1, edgecolor='black', linewidth=0.01, alpha=0.8, label='Actual Data')
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
		finaldata.append([country, finalexp, start + timedelta(days=finalday), start + timedelta(days=when97), maxcases, start + timedelta(days=maxday), mse, mseg, r2, r2g, mape, mapeg])
		plt.xticks(list(range(0,xlim,30)), [(start+timedelta(days=i)).strftime("%b %d") for i in range(0,xlim,skip)], rotation=45, ha='right')
		style = dict( arrowstyle = "-" ,  connectionstyle = "angle", ls =  'dashed')
		annotations = 'cases' if not dead else 'deaths'
		text = plt.annotate('97\% of Total\nPredicted '+annotations+'\non '+(start+timedelta(days=when97)).strftime("%d %b %Y"), xy = ( when97 , 0 ), size='x-small', ha='center', xytext=( when97 , 3*func[whichFunc][0](when97, *popt)), bbox=dict(boxstyle='round', facecolor='white', alpha=0.25), xycoords = 'data' , textcoords = 'data' , fontSize = 16 , arrowprops = style ) 
		text.set_fontsize(10)
		# text2 = plt.annotate('(Gaussian)\n97\% of Total\nPredicted cases\non '+(start+timedelta(days=when97g)).strftime("%d %b %Y"), xy = ( when97g , 0 ), size='x-small', ha='center', xytext=( when97g , 3*func[whichFunc][0](when97, *popt)+4000), bbox=dict(boxstyle='round', facecolor='white', alpha=0.25), xycoords = 'data' , textcoords = 'data' , fontSize = 16 , arrowprops = style ) 
		# text2.set_fontsize(10)
		plt.ylabel("New Cases" if not dead else "New Deaths")
		plt.xlabel("Date")
		plt.legend()
		plt.tight_layout()
		folder = 'newcases' if not dead else 'dead'
		# plt.savefig('graphs/'+folder+'/'+country.replace(" ", "_")+'.pdf')
		print(country)
	except Exception as e:
		print(str(e))
		pass

df = pd.DataFrame(finaldata,columns=['Country','Total','Total Day', '97 Day', 'Max Cases', 'Max Day', 'MSE', 'MSE Gaussian', 'R2', 'R2 Gaussian', 'MAPE', 'MAPE Gaussian'])
df.to_excel('scores.xlsx')
