python3 -m pip install -r requirements.txt


while :
do
	wget https://covid.ourworldindata.org/data/owid-covid-data.csv
	python3 plots.py
	sleep 1d
done