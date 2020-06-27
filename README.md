# Predicting the Growth and Trend of COVID-19 Pandemic

This study applies an improved mathematical model to analyse and predict the growth of the epidemic. An ML-based improved model has been applied to predict the potential threat of COVID-19 in countries worldwide. We show that using iterative weighting for fitting Generalized Inverse Weibull distribution, a better fit can be obtained to develop a prediction framework. This has been deployed on a cloud computing platform for more accurate and real-time prediction of the growth behavior of the epidemic. Interactive prediction graphs can be seen at: https://collaboration.coraltele.com/covid/.

## Quick installation of real-time prediction webapp

To install and run the dynamic real-time prediction webapp on your server run the following commands:
```
$ git clone https://github.com/shreshthtuli/covid-19-prediction.git
$ mv covid-19-prediction covid
$ chmod +x run.sh
$ ./run.sh
```
To access your server go to $HOSTNAME/covid/ from your browser. The webapp is hosted on https://collaboration.coraltele.com/covid2/ where graphs get updated daily based on new data.

## Dataset

We use the <i>[Our World in Data](https://github.com/owid/covid-19-data/tree/master/public/data/)</i> dataset for predicting number of new cases and deaths in various countries.

## Developer

[Shreshth Tuli](https://www.github.com/shreshthtuli) (shreshthtuli@gmail.com)

## Cite this work
If you use our static model, please cite:
```
@article{tuli2020predicting,
title = "Predicting the Growth and Trend of COVID-19 Pandemic using Machine Learning and Cloud Computing",
journal = "Internet of Things",
pages = "100--222",
year = "2020",
issn = "2542-6605",
doi = "https://doi.org/10.1016/j.iot.2020.100222",
url = "http://www.sciencedirect.com/science/article/pii/S254266052030055X",
author = "Shreshth Tuli and Shikhar Tuli and Rakesh Tuli and Sukhpal Singh Gill",
}
```
If you use our dynamic model, please cite:
```
@article{tuli2020modelling,
  title={Modelling for prediction of the spread and severity of COVID-19 and its association with socioeconomic factors and virus types},
  author={Tuli, Shreshth and Tuli, Shikhar and Verma, Ruchi and Tuli, Rakesh},
  journal={medRxiv},
  year={2020},
  publisher={Cold Spring Harbor Laboratory Press}
}
```

## References
* **Shreshth Tuli, Shikhar Tuli, Rakesh Tuli and Sukhpal Singh Gill, [Predicting the Growth and Trend of COVID-19 Pandemic using Machine Learning and Cloud Computing.](https://www.sciencedirect.com/science/article/pii/S254266052030055X?via%3Dihub) Internet of Things, ISSN: 2542-6605, Elsevier Press, Amsterdam, The Netherlands, May 2020.** ([Open access link](https://www.medrxiv.org/content/10.1101/2020.05.06.20091900v1))
* **Shreshth Tuli, Shikhar Tuli, Ruchi Verma and Rakesh Tuli, [Modelling for prediction of the spread and severity of COVID-19 and its association with socioeconomic factors and virus types.](https://www.medrxiv.org/content/10.1101/2020.06.18.20134874v1) medRxiv, June 2020.** ([Open access link](https://www.medrxiv.org/content/10.1101/2020.06.18.20134874v1))
