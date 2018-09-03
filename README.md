# hydroutils

`hydroutils` is a small python library to solve common problems in Hydrology

### Current features

- Annual maxima estimation
- Statistical distribution fitting
- Distribution parameters estimation based on L-moments or MLE

### Installation

Just run this line

``` pip install git+https://github.com/jsosa/hydroutils.git```

### Dependencies

- [Pandas](https://pandas.pydata.org/)
- [statsmodels](https://www.statsmodels.org/stable/index.html)
- [scipy](https://www.scipy.org/)

### Usage

The module is called via

```python
import hydroutils.core as hu
```

`hydroutils` uses a `pd.DataFrame` object with indexes using a `pd.DatetimeIndex` object:

![pic01](images/pic01.png?raw=true)

Annual maxima (AMAX) is calculated via:

```python
hu.find_events_amax(serie)
```

![pic02](images/pic02.png?raw=true)

where `serie` is the `pd.DataFrame` object defined before

AMAX can be visualized via:

```python
ax = serie.plot()
amax.plot(ax=ax,style='o',c='Red')
ax.set_xlabel('Date')
ax.set_ylabel('Discharge (m^3/s)')
```

![pic03](images/pic03.png?raw=true)

A plot showing a fitted distribution can be obtained via:

```python
amax_vals = np.sort(amax.values.squeeze())[1:]
ax = hu.plot_return(amax_vals,hu.stats.pearson3,'mle')
```

![pic04](images/pic04.png?raw=true)

where `amax_vals` is the `pd.DataFrame` object obtained in the previous step `hu.stats.pearson3` can be replaced by any other distribution from Scipy, check a list of distributions available here <https://docs.scipy.org/doc/scipy/reference/stats.html>. `mle` is the method used to estimate the distribution parameters. Optionally `lmo` can also be used to estimate parameters by using original fortran J.R.M Hosking subroutines.

In hydrology is preferred to work with Return Periods, then a function has been created to retreive corresponding discharge value to a given return period

Retrive discharge for a given return period can be done via:

```python
hu.get_dis_rp(amax.values,hu.stats.pearson3,ret,'mle')
```

where `amax_vals` is the `pd.DataFrame` object containing annual maxima, `hu.stats.pearson3` is a statistical distribution from Scipy, `ret` is a return period for example 100 for a 100-yr return period and `mle` is the method used to estimate parameters in the distribution.