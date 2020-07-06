import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.metrics import mean_squared_error
import numpy as np
from copy import deepcopy
from numpy import inf
from math import exp
from datetime import timedelta
from sklearn.metrics import r2_score
from scipy.special import softmax
import warnings
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.graph_objects import Layout
from datetime import datetime, date

warnings.simplefilter("ignore")

# plt.style.use(['science'])
# plt.rcParams["text.usetex"] = True

try:
    os.makedirs('plots')
except OSError as e:
	pass

filename = 'owid-covid-data.csv'
if not os.path.exists(filename): filename = 'owid-covid-data-bak.csv'
df = pd.read_csv(filename)
df['date'] = pd.to_datetime(df.date)

countries = list(pd.unique(df['location']))

def weib(x, k, a, b, g):
	return k * g * b * (a ** b) * np.exp(-1 * g * ((a / x)  ** b)) / (x ** (b + 1))

def getInfoCountry(df2, isdead):
	df2['Delta'] = (df2.date - min(df2.date)).dt.days
	startDate = min(df2.date)
	totalLength = max(df2.Delta)
	confirmed = []; new = []
	df2 = df2.fillna(0)
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
		outliersweight = outliersweight - np.tanh(outliersweight)
		outliersweight = outliersweight / np.max(outliersweight)
		outliersweight = softmax(1 - outliersweight)
		if i > 1 and sum(abs(old - outliersweight)) < 0.001: break
	return popt, pcov

def seriesIterativeCurveFit(func, xIn, yIn, start):
	res = []; errors = []
	for ignore in range(10, 0, -1):
		x = xIn[:-1*ignore]; y = yIn[:-1*ignore]
		outliersweight = None
		for i in range(10):
			try:
				popt, pcov = curve_fit(func, x, y, start, sigma=outliersweight, absolute_sigma=True, maxfev=500000)
			except Exception as e: 
				print('ignore -', ignore, ', exception -', e)
				break
			pred = np.array([func(px, *popt) for px in x])
			old = outliersweight
			outliersweight = np.abs(pred - y)
			outliersweight = outliersweight / np.max(outliersweight)
			outliersweight = outliersweight - np.tanh(outliersweight)
			outliersweight = softmax(1 - outliersweight)
			if i > 1 and sum(abs(old - outliersweight)) < 0.001: break
		pred = [func(px, *popt) for px in xIn]
		res.append((popt, pcov, ignore))
		errors.append(mean_absolute_percentage_error(yIn, pred))
	print(errors)
	val = res[errors.index(min(errors))]
	return val[0], val[1]

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
dfcPlot = pd.DataFrame()
training_data = -1
interactive = ['India', 'World', 'United States', 'United Kingdom', 'Brazil', 'Italy', 'France', 'Germany', 'Russia']
for country in interactive:
	try:
		dead = False
		print("--", country)
		df2 = df[df['location'] == country]
		res = getInfoCountry(df2, False)
		data = res[-1]
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
		cumsave = getcummulative(data)

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

		cumdpredsave = getcummulative(deadpredsave)
		cumdsave = getcummulative(deadsave)
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
		newcdf = pd.DataFrame.from_dict({country+'-dates':[(start+timedelta(days=i)).strftime("%d %b %Y") for i in list(range(1,xlim))], \
			country+'-cpred': cumpredsave, \
			country+'-ctrue': cumsave,\
			country+'-cpredd': cumdpredsave,\
			country+'-ctrued': cumdsave}, orient='index').T
		dfcPlot[country+'-dates'] = pd.Series(newcdf[country+'-dates'])
		dfcPlot[country+'-pred'] = pd.Series(newcdf[country+'-cpred'])
		dfcPlot[country+'-true'] = pd.Series(newcdf[country+'-ctrue'])
		dfcPlot[country+'-cpredd'] = pd.Series(newcdf[country+'-cpredd'])
		dfcPlot[country+'-ctrued'] = pd.Series(newcdf[country+'-ctrued'])
		dates = [(start+timedelta(days=i)).strftime("%d %b %Y") for i in list(range(1,xlim))]
		fig = make_subplots(specs=[[{"secondary_y": True}]])
		fig.add_trace(go.Scatter(x=dates, y=newpredsave, name="Prediction (new cases)", marker=dict(color='#EB752C')))
		fig.add_trace(go.Bar(x=dates, y=newsave, name="True data (new cases)", marker=dict(color='#EB752C'), opacity=0.7, width=[1]*len(dates), hoverlabel=dict(bgcolor='#FF351C')))
		fig.add_trace(go.Scatter(x=dates, y=deadpredsave, name="Prediction (deaths)", marker=dict(color='#2D58BE')), secondary_y=True)
		fig.add_trace(go.Bar(x=dates, y=deadsave, name="True data (deaths)", marker=dict(color='#2D58BE'), opacity=0.3, width=[1]*len(dates), hoverlabel=dict(bgcolor='#1A22AB')), secondary_y=True)
		fig.update_layout(hovermode="x",
			title=country.capitalize(),
			title_x=0.5,
			title_font=dict(size=20),
			xaxis_title='Date',
			font=dict(family='Overpass', size=12, color='#212121'),
			yaxis_tickformat = ',.0f',
			xaxis = dict(dtick = 30),
			autosize=False,
		    width=700,
		    height=500,
		    legend=dict(x=0.6, y=0.9, bordercolor='Black', borderwidth=1),
		    plot_bgcolor='rgba(0,0,0,0)',
			)
		fig.update_yaxes(title_text="Number of daily new cases", gridcolor='lightgray', showline=True, linewidth=3, linecolor='orange', gridwidth=1, secondary_y=False,)
		fig.update_yaxes(tickformat = ',.0f',title_text="Number of daily deaths", secondary_y=True, gridcolor='lightgray', showline=True, linewidth=3, linecolor='blue', gridwidth=1)
		fig.write_html("plots/"+country+"_pred"+".html")
		####
		fig = make_subplots(specs=[[{"secondary_y": True}]])
		fig.add_trace(go.Scatter(x=dates, y=cumpredsave, name="Prediction (new cases)", marker=dict(color='#EB752C')))
		fig.add_trace(go.Bar(x=dates, y=cumsave, name="True data (new cases)", marker=dict(color='#EB752C'), opacity=0.7, width=[1]*len(dates), hoverlabel=dict(bgcolor='#FF351C')))
		fig.add_trace(go.Scatter(x=dates, y=cumdpredsave, name="Prediction (deaths)", marker=dict(color='#2D58BE')), secondary_y=True)
		fig.add_trace(go.Bar(x=dates, y=cumdsave, name="True data (deaths)", marker=dict(color='#2D58BE'), opacity=0.3, width=[1]*len(dates), hoverlabel=dict(bgcolor='#1A22AB')), secondary_y=True)
		fig.update_layout(hovermode="x",
			title=country.capitalize(),
			title_x=0.5,
			title_font=dict(size=20),
			xaxis_title='Date',
			font=dict(family='Overpass', size=12, color='#212121'),
			yaxis_tickformat = ',.0f',
			xaxis = dict(dtick = 30),
			autosize=False,
		    width=700,
		    height=500,
		    legend=dict(x=0.6, y=0.3, bordercolor='Black', borderwidth=1),
		    plot_bgcolor='rgba(0,0,0,0)',
			)
		fig.update_yaxes(title_text="Number of total cases", gridcolor='lightgray', showline=True, linewidth=3, linecolor='orange', gridwidth=1, secondary_y=False,)
		fig.update_yaxes(tickformat = ',.0f',title_text="Number of total deaths", secondary_y=True, gridcolor='lightgray', showline=True, linewidth=3, linecolor='blue', gridwidth=1)
		fig.write_html("plots/"+country+"_total"+".html")
		fig.data = []
	except Exception as e:
		print(str(e))
		# raise(e)
		pass

f = open("plots/lastupdated.txt", "w")
now = datetime.now().time()
today = date.today()
f.write(now.strftime("%H:%M:%S")+" - "+today.strftime("%d %b %Y"))
f.close()
# dfPlot.to_excel('plot.xlsx')
# dfcPlot.to_excel('cplot.xlsx')
