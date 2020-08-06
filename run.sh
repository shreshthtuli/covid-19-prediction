python3 -m pip install -r requirements.txt
sudo apt install python3-tk

while :
do
	wget -O owid-covid-data.csv https://covid.ourworldindata.org/data/owid-covid-data.csv
	python3 plots.py
	sleep 1d
done