# Predicting the Growth and Trend of COVID-19 Pandemic

This study applies an improved mathematical model to analyse and predict the growth of the epidemic. An ML-based improved model has been applied to predict the potential threat of COVID-19 in countries worldwide. We show that using iterative weighting for fitting Generalized Inverse Weibull distribution, a better fit can be obtained to develop a prediction framework. This has been deployed on a cloud computing platform for more accurate and real-time prediction of the growth behavior of the epidemic.

## Dataset

We use the <i>[Our World in Data](https://github.com/owid/covid-19-data/tree/master/public/data/)</i> dataset for predicting number of new cases and deaths in various countries.

<div>
    <a href="https://plotly.com/~shreshthtuli/6/?share_key=RaWuX710gE2BbdFGiA5WEc" target="_blank" title="India" style="display: block; text-align: center;"><img src="https://plotly.com/~shreshthtuli/6.png?share_key=RaWuX710gE2BbdFGiA5WEc" alt="India" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="shreshthtuli:6" sharekey-plotly="RaWuX710gE2BbdFGiA5WEc" src="https://plotly.com/embed.js" async></script>
</div>

<div>
    <a href="https://plotly.com/~shreshthtuli/4/?share_key=RaWuX710gE2BbdFGiA5WEc" target="_blank" title="World" style="display: block; text-align: center;"><img src="https://plotly.com/~shreshthtuli/4.png?share_key=RaWuX710gE2BbdFGiA5WEc" alt="World" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="shreshthtuli:4" sharekey-plotly="RaWuX710gE2BbdFGiA5WEc" src="https://plotly.com/embed.js" async></script>
</div>

<div>
    <a href="https://plotly.com/~shreshthtuli/5/?share_key=RaWuX710gE2BbdFGiA5WEc" target="_blank" title="United States" style="display: block; text-align: center;"><img src="https://plotly.com/~shreshthtuli/5.png?share_key=RaWuX710gE2BbdFGiA5WEc" alt="United States" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="shreshthtuli:5" sharekey-plotly="RaWuX710gE2BbdFGiA5WEc" src="https://plotly.com/embed.js" async></script>
</div>

<div>
    <a href="https://plotly.com/~shreshthtuli/7/?share_key=RaWuX710gE2BbdFGiA5WEc" target="_blank" title="United Kingdom" style="display: block; text-align: center;"><img src="https://plotly.com/~shreshthtuli/7.png?share_key=RaWuX710gE2BbdFGiA5WEc" alt="United Kingdom" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="shreshthtuli:7" sharekey-plotly="RaWuX710gE2BbdFGiA5WEc" src="https://plotly.com/embed.js" async></script>
</div>

<div>
    <a href="https://plotly.com/~shreshthtuli/8/?share_key=RaWuX710gE2BbdFGiA5WEc" target="_blank" title="China" style="display: block; text-align: center;"><img src="https://plotly.com/~shreshthtuli/8.png?share_key=RaWuX710gE2BbdFGiA5WEc" alt="China" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="shreshthtuli:8" sharekey-plotly="RaWuX710gE2BbdFGiA5WEc" src="https://plotly.com/embed.js" async></script>
</div>

<div>
    <a href="https://plotly.com/~shreshthtuli/9/?share_key=RaWuX710gE2BbdFGiA5WEc" target="_blank" title="Spain" style="display: block; text-align: center;"><img src="https://plotly.com/~shreshthtuli/9.png?share_key=RaWuX710gE2BbdFGiA5WEc" alt="Spain" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="shreshthtuli:9" sharekey-plotly="RaWuX710gE2BbdFGiA5WEc" src="https://plotly.com/embed.js" async></script>
</div>

<div>
    <a href="https://plotly.com/~shreshthtuli/10/?share_key=RaWuX710gE2BbdFGiA5WEc" target="_blank" title="Italy" style="display: block; text-align: center;"><img src="https://plotly.com/~shreshthtuli/10.png?share_key=RaWuX710gE2BbdFGiA5WEc" alt="Italy" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="shreshthtuli:10" sharekey-plotly="RaWuX710gE2BbdFGiA5WEc" src="https://plotly.com/embed.js" async></script>
</div>


<div>
    <a href="https://plotly.com/~shreshthtuli/11/?share_key=RaWuX710gE2BbdFGiA5WEc" target="_blank" title="France" style="display: block; text-align: center;"><img src="https://plotly.com/~shreshthtuli/11.png?share_key=RaWuX710gE2BbdFGiA5WEc" alt="France" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="shreshthtuli:11" sharekey-plotly="RaWuX710gE2BbdFGiA5WEc" src="https://plotly.com/embed.js" async></script>
</div>


<div>
    <a href="https://plotly.com/~shreshthtuli/12/?share_key=RaWuX710gE2BbdFGiA5WEc" target="_blank" title="Germany" style="display: block; text-align: center;"><img src="https://plotly.com/~shreshthtuli/12.png?share_key=RaWuX710gE2BbdFGiA5WEc" alt="Germany" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="shreshthtuli:12" sharekey-plotly="RaWuX710gE2BbdFGiA5WEc" src="https://plotly.com/embed.js" async></script>
</div>

<div>
    <a href="https://plotly.com/~shreshthtuli/13/?share_key=RaWuX710gE2BbdFGiA5WEc" target="_blank" title="Russia" style="display: block; text-align: center;"><img src="https://plotly.com/~shreshthtuli/13.png?share_key=RaWuX710gE2BbdFGiA5WEc" alt="Russia" style="max-width: 100%;width: 600px;"  width="600" onerror="this.onerror=null;this.src='https://plotly.com/404.png';" /></a>
    <script data-plotly="shreshthtuli:13" sharekey-plotly="RaWuX710gE2BbdFGiA5WEc" src="https://plotly.com/embed.js" async></script>
</div>

## Developer

[Shreshth Tuli](https://www.github.com/shreshthtuli) (shreshthtuli@gmail.com)

## Cite this work
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

## References
* **Shreshth Tuli, Shikhar Tuli, Rakesh Tuli and Sukhpal Singh Gill, [Predicting the Growth and Trend of COVID-19 Pandemic using Machine Learning and Cloud Computing.](https://www.sciencedirect.com/science/article/pii/S254266052030055X?via%3Dihub) Internet of Things, ISSN: 2542-6605, Elsevier Press, Amsterdam, The Netherlands, May 2020.** ([Open access link](https://www.medrxiv.org/content/10.1101/2020.05.06.20091900v1))
