import os
import pandas as pd
from shutil import copyfile

df = pd.read_excel('../results/correlation.xlsx', sheet_name='Raw Data >0.8')
goodCountries = list(pd.unique(df['Country']))
goodCountries = [i.replace(" ", "_") for i in goodCountries]

# os.mkdir('good')
for i in goodCountries:
	try:
		copyfile('both/'+i+'.pdf', 'good/'+i+'.pdf')
	except Exception as e:
		print(i)
