import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import timedelta

plt.style.use(['science'])
plt.rcParams["text.usetex"] = True

pop = pd.read_csv('population_india_census2011.csv')
pop_dict = dict(zip(pop['State / Union Territory'], [float(i.split('/')[0].replace(',', '')) for i in pop['Density']]))
total_dict = dict(zip(pop['State / Union Territory'], [float(i) for i in pop['Population']]))

beds = pd.read_csv('HospitalBedsIndia.csv')
beds_dict = dict(zip(beds['State/UT'], [float(i.replace(',', '') if 'str' in str(type(i)) else i) for i in beds['NumPublicBeds_HMIS']]))

df = pd.read_csv('covid_19_india.csv')
df['Date'] = pd.to_datetime(df.Date, dayfirst=True)

states = list(pd.unique(df['State/UnionTerritory']))

def func(x, a, b, c):
    return a * np.exp(b * x) + c 

startDate = min(df.Date)
print(startDate)

month2 = startDate + timedelta(days=80)
month4 = startDate + timedelta(days=90)
month6 = startDate + timedelta(days=120)
print(month4)

total = 0

s2 = ['Delhi',
'Maharashtra',
'Telengana',
'Jammu and Kashmir',
'Tamil Nadu',
'Kerala',
'Rajasthan',
'Himachal Pradesh',
'Andhra Pradesh',
'Chandigarh',
'Haryana',
'Madhya Pradesh',
'Gujarat',
'Chattisgarh',
'Uttarakhand',
'Punjab',
'Uttar Pradesh',
'Karnataka',
'Odisha',
'West Bengal',
'Ladakh',
'Pondicherry',
'Bihar'
]

for state in s2:
	df2 = df[df['State/UnionTerritory']==state]
	df2['Delta'] = df2.Date - min(df2.Date)
	# print(df2[['Date', 'ConfirmedIndianNational', 'Delta']]); exit()
	x = np.array(df2.Delta/(24*60*60*1000000000), dtype=float); yn = np.array(df2.Confirmed)
	try: 
		# print(state)
		# print(beds_dict['Jammu and Kashmir' if state == 'Ladakh' else state])
		# print(total_dict['Jammu and Kashmir' if state == 'Ladakh' else state])
		popt, pcov = curve_fit(func, x, yn)
		px = (month2-min(df2.Date)).days
		# print(px)
		print(func(px, *popt))
		if state != "Kerala": total += func(px, *popt)
	except Exception as e: 
		# print(beds_dict['Jammu and Kashmir' if state == 'Ladakh' else state])
		# print(total_dict['Jammu and Kashmir' if state == 'Ladakh' else state])
		# print(state, str(e))
		# pass
		continue
	if state == "Delhi":
		plt.figure(figsize=(5,4))
		plt.plot(x, yn, 'ko', label="Original Confirmed cases Data")
		plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
		dt = list(df2.Date)
		plt.xticks(list(range(0,len(x),3)), [dt[i].strftime("%b %d") for i in range(0,len(dt),3)], rotation=45, ha='right')
		plt.xlabel("Date")
		plt.ylabel("Confirmed Cases")
		plt.tight_layout()
		plt.legend()
		plt.savefig('delhi.pdf')

print(total)