import numpy as np
from collections import namedtuple

import pandas as pd
import matplotlib.pylab as plt
from datetime import datetime


def __preprocessing(x):
    try:
        if x.index.dtype != 'int64':
            idx = x.index.date.astype('str')
        else:
            idx = np.asarray(range(1, len(x)+1))
    except:
        idx = np.asarray(range(1, len(x)+1))
        
    x = np.asarray(x)
    dim = x.ndim
    
    if dim == 1:
        c = 1
        
    elif dim == 2:
        (n, c) = x.shape
        
        if c == 1:
            dim = 1
            x = x.flatten()
            
    else:
        print('Please check your dataset.')
        
    return x, c, idx

def __missing_values_analysis(x, idx, method = 'skip'):
    if method.lower() == 'skip':
        if x.ndim == 1:
            idx = idx[~np.isnan(x)]
            x = x[~np.isnan(x)]
            
        else:
            idx = idx[~np.isnan(x).any(axis=1)]
            x = x[~np.isnan(x).any(axis=1)]
            
    n = len(x)
    
    return x, n, idx

def ordered_cluster_analysis(x):
    res = namedtuple('Ordered_Cluster_Analysis', ['cp', 'S','idx'])
    x, c, idx = __preprocessing(x)
    x, n, idx = __missing_values_analysis(x, idx, method = 'skip')
    S = []
    for t in range(1,len(x)):
        _x_t = sum(x[:t])/len(x[:t])
        _x_n_t = sum(x[t:])/len(x[t:])
        V1 = sum([(val-_x_t)**2 for val in x[:t]])
        V2 = sum([(val-_x_n_t)**2 for val in x[t:]])
        S.append(V1+V2)

    cp_idx = np.argmin(S)
    cp = idx[cp_idx]
    
    return res(cp,S,idx[:-1])

def plot_abrupt(x,**kwargs):
    """
    Parameters:
    * `x` ['pandas Series']
        a time series
    * `series_name` [string, optional]:
        The series name, if known.
    
    * `ax` [`Axes`, optional]:
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    * `fig_id` [string, optional]:
        The index of the ax, if known.

    return
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """

    ax = kwargs.get('ax',None)
    series_name = kwargs.get('series_name', None)
    fig_id = kwargs.get('fig_id', None)

    if ax is None:
        ax = plt.gca()
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y',useMathText=True)

    cp,S,idx = ordered_cluster_analysis(x)
    if type(cp)!=np.int32 or type(cp)!=np.int64:
        cp = datetime.strptime(cp, '%Y-%m-%d').year
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in idx]
        idx = [date.year for date in dates]

    if series_name is not None:
        ax.plot(idx,S,c='b',label=series_name)
    else:
        ax.plot(idx,S,c='b')

    
    ax.axvline(x=cp, color='r', linestyle='--', label='Mutation('+str(cp)+')')
    ax.set_xlabel('Date (Year)')
    # add text around the vertical line to indicate the date of change point
    # ax.text(datetime.strptime(cp, '%Y-%m-%d').year, max(S), datetime.strptime(cp, '%Y-%m-%d').year, color='r', fontsize=12, ha='center', va='bottom')
    
    # if fig_id is None:
    #     ax.set_xlabel('Date (Year)', )
    # else:
    #     ax.set_xlabel('Date (Year)\n('+fig_id+')', )
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_pos = xlim[0] + 0.02 * (xlim[1] - xlim[0])
    y_pos = ylim[0] + 0.02 * (ylim[1] - ylim[0])
    if fig_id is not None:
        ax.text(x_pos, y_pos, fig_id, ha='left', va='bottom')


    ax.set_ylabel('S')
    ax.set_title('Ordered Cluster Analysis')
    ax.legend(loc='lower right',shadow=False,frameon=False)
    
    return ax


    

   


if __name__ == '__main__':
    xunhua = pd.read_csv('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/data/xunhua_annual_runoff.csv',parse_dates=['date'],index_col='date')
    guide = pd.read_csv('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/data/guide_annual_runoff.csv',parse_dates=['date'],index_col='date')
    # print(type(xunhua['runoff(10^8m^3)']))
    # res = ordered_cluster_analysis(xunhua)
    # print(res)
    # plt.plot(S)
    # plt.show()
    # xunhua['year'] = xunhua.index.year
    # guide['year'] = guide.index.year
    # xunhua.set_index('year',inplace=True)
    # guide.set_index('year',inplace=True)

    # print(xunhua.index.dtype)

    fig = plt.figure(figsize=(7.48,4.5))
    ax1=fig.add_subplot(1,2,1)
    plot_abrupt(guide['runoff(10^8m^3)'].values,ax=ax1,series_name='Guide')
    ax2=fig.add_subplot(1,2,2)
    plot_abrupt(xunhua['runoff(10^8m^3)'],ax=ax2,series_name='Xunhua')
    plt.show()