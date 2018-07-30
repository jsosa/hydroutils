#!/usr/bin/env python

# inst: university of bristol
# auth: jeison sosa
# mail: j.sosa@bristol.ac.uk / sosa.jeison@gmail.com

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sys import exit
from scipy import stats
from hydroutils.lmoments.SAMLMU import samlmu
from hydroutils.lmoments.PELEXP import pelexp
from hydroutils.lmoments.PELGAM import pelgam
from hydroutils.lmoments.PELGEV import pelgev
from hydroutils.lmoments.PELGLO import pelglo
from hydroutils.lmoments.PELGNO import pelgno
from hydroutils.lmoments.PELGPA import pelgpa
from hydroutils.lmoments.PELGUM import pelgum
from hydroutils.lmoments.PELKAP import pelkap
from hydroutils.lmoments.PELNOR import pelnor
from hydroutils.lmoments.PELPE3 import pelpe3
from hydroutils.lmoments.PELWAK import pelwak

def _check_temporal(df,days):
    
    df = df.copy()
    val = df.iloc[:,0].values
    idx = df.index

    res_idx = []
    res_val = []
    first = idx[0]
    block_idx = []
    block_val = []
    
    for i in range(idx.size):
        dif = idx[i]-first
        if dif.days <= days:
            block_idx.append(idx[i])
            block_val.append(val[i])
        else:
            ix = np.argmax(block_val)
            res_idx.append(block_idx[ix])
            res_val.append(block_val[ix])
            first = idx[i]
            block_idx = []
            block_val = []
            block_idx.append(idx[i])
            block_val.append(val[i])
        
        # We should calculate maximum for ast block
        if i == idx.size-1:
            ix = np.argmax(block_val)
            res_idx.append(block_idx[ix])
            res_val.append(block_val[ix])

    df_res = pd.DataFrame({df.keys()[0]:res_val, 'time':res_idx})
    df_res.set_index(df_res['time'],inplace=True)
    del df_res['time']

    return df_res

def _check_quantile(df,qua):

    df = df.copy()
    df = _get_quantile(df,qua)
    df['lt_qua'] = df[df.keys()[0]]>=df['threshold']
    dis = df.where(df['lt_qua']).dropna().drop(['threshold','lt_qua'],axis=1)

    return dis

def _get_quantile(df,qua_val):

    df = df.copy()
    # df['threshold'] = df.groupby(df.index.year).transform(lambda x: x.quantile(qua_val))
    df['threshold'] = df.quantile(qua_val).values[0]

    return df

def _get_weibull_return(data):

    # df   = data.copy()
    # df   = df.sort_values('discharge')
    # leng = df['discharge'].size + 1
    # df['rank']   = np.arange(1,leng)
    # df['ANEP']   = df['rank']/(leng)
    # df['AEP']    = 1-df['ANEP']
    # df['return'] = 1/df['AEP']

    dis     = np.sort(data)
    leng    = dis.size + 1
    rank    = np.arange(1,leng)
    ANEP    = rank/leng
    AEP     = 1-ANEP
    returnp = 1/AEP

    return returnp, dis

def _prep_test_return(data,dist):

    x,y = _get_weibull_return(data)

    # x = x[-5:]
    # y = y[-5:]
    df = pd.DataFrame({'return':x, 'real':y})

    model = []
    for ret in x:
        val = get_dis_rp(data,dist,ret)
        model.append(val)
    df['model']=model

    return df

def _test_return(data,dist):

    df = _prep_test_return(data,dist)
    df['diff'] = abs(df['model']-df['real'])
    value = df['diff'].sum()
    
    return value

def automatic_fitting(data):

    distributions = [
                    stats.expon,
                    stats.gamma,
                    stats.genextreme,
                    stats.genlogistic,
                    stats.genpareto,
                    stats.gennorm,
                    stats.gumbel_r,
                    stats.kappa4,
                    stats.norm,
                    stats.pearson3
                    ]

    p = []
    for dist in distributions:
        val = _test_return(data,dist)
        p.append(val)

    i = np.argmin(p)
    
    return distributions[i]

def find_events(df,qua=0.99,days=7):

    df = df.copy()
    dis0 = _check_quantile(df,qua=qua)
    dis  = _check_temporal(dis0,days=days)
    return dis

def get_params(array,dist,params):

    arr = array
    if params == 'mle':
        return dist.fit(arr)
    elif params == 'lmo':
        if dist.name == 'expon':
            xmom = samlmu(arr,arr.size,2)
            a = pelexp(xmom)
            return (a[1],a[0])
        elif dist.name == 'gamma':
            xmom = samlmu(arr,arr.size,2)
            a = pelgam(xmom)
            return (a[1],a[0])
        elif dist.name == 'genextreme':
            xmom = samlmu(arr,arr.size,3)
            a = pelgev(xmom)
            return (a[2],a[0],a[1])
        elif dist.name == 'genlogistic':
            xmom = samlmu(arr,arr.size,3)
            a = pelglo(xmom)
            return (a[2],a[0],a[1])
        elif dist.name == 'gennorm':
            xmom = samlmu(arr,arr.size,3)
            a = pelgno(xmom)
            return (a[2],a[0],a[1])
        elif dist.name == 'genpareto':
            xmom = samlmu(arr,arr.size,3)
            a = pelgpa(xmom)
            return (a[2],a[0],a[1])
        elif dist.name == 'gumbel_r':
            xmom = samlmu(arr,arr.size,2)
            a = pelgum(xmom)
            return (a[0],a[1])
        elif dist.name == 'kappa4':
            xmom = samlmu(arr,arr.size,4)
            a,b = pelkap(xmom)
            return (a[3],a[2],a[1],a[0])
        elif dist.name == 'norm':
            xmom = samlmu(arr,arr.size,2)
            a = pelnor(xmom)
            return (a[0],a[1])
        elif dist.name == 'pearson3':
            xmom = samlmu(arr,arr.size,3)
            a = pelpe3(xmom)
            return (a[2],a[0],a[1])
        elif dist.name == 'wakeby':
            xmom = samlmu(arr,arr.size,5)
            a = pelwak(xmom)
            return (a[4],a[3],a[2],a[0],a[1])
        else:
            exit('ERROR L-moments not implemented for such distribution')
    else:
        exit('ERROR valid parameters estimation are mle or lmo')

def get_dis_rp(array,dist,rp,params):
    
    arr = array
    fit = get_params(array,dist,params)
    ppf = dist.ppf(1-1/rp,*fit)

    return ppf

def plot_quant_probs(data,dist):

    a = sm.ProbPlot(data,dist=dist,fit=True)
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    aplt = a.qqplot(ax=ax,line='45')
    ax = fig.add_subplot(1, 2, 2)
    aplt = a.ppplot(ax=ax,line='45')

def plot_cdf(data,dist):

    fit = dist.fit(data)
    
    ecdf = sm.distributions.ECDF(data)
    ax = plt.plot(ecdf.x, ecdf.y,'.',label='empirical')
    
    xx = np.linspace(data.min(),data.max(),100)
    yy = dist.cdf(xx,*fit)
    ax = plt.plot(xx,yy,label=dist.name)
    
    plt.legend(loc='right')
    plt.ylabel('Probability')
    plt.xlabel('Value')
    
def plot_return(data,dist,params):
    
    x,y = _get_weibull_return(data)
    
    xx  = np.linspace(0,100,10000)
    fit = get_params(data,dist,params)
    yy  = dist.ppf(1-1/xx,*fit)

    ax  = plt.figure(figsize=(4.5,4.5))
    ax  = plt.plot(x,y,'.',label='amax')
    ax  = plt.plot(xx,yy,label=dist.name,color='Red')
    
    plt.xscale('log')
    plt.legend(loc='right')
    plt.xlabel('Return Period (years)')
    plt.ylabel('Discharge (m^3/s)')

    return ax

def find_events_amax(data):

    df   = data.copy()
    vals = df.groupby(df.index.year).max()
    tims = df.groupby(df.index.year).idxmax()
    eve  = pd.DataFrame(vals)
    key  = eve.keys()[0]
    eve.set_index(tims[key],inplace=True) 

    return eve

def plot_events(data,method,qua_val=0.7,days_val=7):

    df = data.copy()

    if method == 'pot':
        qua = _get_quantile(df,qua_val=qua_val)
        eve = find_events(df,qua=qua_val,days=days_val)
        key = eve.keys()[0]
        eve.rename({key:'events'},axis=1,inplace=True)
        ax = qua.plot()
        eve.plot(ax=ax,style='.k')
        plt.ylabel('Value')
    elif method == 'amax':
        eve = find_events_amax(df)
        key = eve.keys()[0]
        eve.rename({key:'events'},axis=1,inplace=True)
        ax = df.plot()
        eve.plot(ax=ax,style='.k')
        plt.ylabel('Value')
    else:
        exit('ERROR method not recognized')

def auto_fit(sample):

    cdfs = [
            "expon",
            "gamma",
            "genextreme",
            "genlogistic",
            "genpareto",
            "gennorm",
            "gumbel_r",
            "kappa4",
            "norm",
            "pearson3"
            ]

    df        = pd.DataFrame()
    df['p']   = np.nan
    df['D']   = np.nan
    df['cdf'] = cdfs

    for i in range(len(cdfs)):

        # Fit our data set against every probability distribution
        parameters = eval("stats."+cdfs[i]+".fit(sample)");

        # Applying the Kolmogorov-Smirnof one sided test
        D, p = stats.kstest(sample, cdfs[i], args=parameters);

        df.loc[i,'p'] = p
        df.loc[i,'D'] = D

        best_cdf = df.loc[df['D'] == df.D.min(),'cdf'].values[0]
        dist     = eval("stats." + best_cdf)
        
    return dist
