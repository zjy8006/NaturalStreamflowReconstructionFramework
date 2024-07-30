import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
# plt.rcParams['figure.figsize']=(10,8)
plt.rcParams['font.size']=6
# plt.rcParams["figure.figsize"] = [12, 8]
# plt.rcParams['image.cmap']='plasma'
# plt.rcParams['axes.linewidth']=0.8
import math
from scipy.stats import norm
from collections import namedtuple
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

def get_Sk(x):
    res = namedtuple('Sk', ['Sk','idx'])
    x, c, idx = __preprocessing(x)
    x, n, idx = __missing_values_analysis(x, idx, method = 'skip')
    Sk = []
    for k in range(1, len(x)): #(2,N)
        sumss = []
        for i in range(k+1):
            aij = []
            for j in range(i+1):
                val = x[i]-x[j]
                if val > 0:
                    aij.append(1)
                else:
                    aij.append(0)
            sumss.append(sum(aij))
        Sk.append(sum(sumss))
    return res(Sk,idx[:-1])

def get_ESk(x):
    res = namedtuple('ESk', ['ESk','idx'])
    x, c, idx = __preprocessing(x)
    x, n, idx = __missing_values_analysis(x, idx, method = 'skip')
    ESk = []
    for k in range(2, len(x)+1):
        ESk.append(k*(k-1)/4.0)
    return res(ESk,idx[:-1])

def get_var_Sk(x):
    res = namedtuple('var_Sk', ['var_Sk','idx'])
    x, c, idx = __preprocessing(x)
    x, n, idx = __missing_values_analysis(x, idx, method = 'skip')
    var_Sk = []
    for k in range(2, len(x)+1):
        var_Sk.append(k*(k-1)*(2*k+5)/72.0)
    return res(var_Sk,idx[:-1])

def get_UFk(x):
    res = namedtuple('UFk', ['UFk','idx'])
    Sk,idx = get_Sk(x)
    ESk,idx = get_ESk(x)
    var_Sk,idx = get_var_Sk(x)
    UFk = []
    for i in range(len(Sk)):
        UFk.append((Sk[i]-ESk[i])/math.sqrt(var_Sk[i]))
    return res(UFk,idx)

def get_UBk(x):
    res = namedtuple('UBk', ['UBk','idx'])
    x, c, idx = __preprocessing(x)
    x, n, idx = __missing_values_analysis(x, idx, method = 'skip')
    reverse_series = list(reversed(x))
    Sk,idx_ = get_Sk(reverse_series)
    ESk,idx_ = get_ESk(reverse_series)
    var_Sk,idx_ = get_var_Sk(reverse_series)
    UBk = []
    for i in range(len(Sk)):
        UBk.append((Sk[i]-ESk[i])/math.sqrt(var_Sk[i]))
    return res(UBk,idx)


def get_Z_alpha(confidence):
    var = 1-(1-confidence)/2
    return norm._ppf(var)



def plot_abrupt(x, **kwargs):
    """ Plot abrupt of a time series using Mann-Kendall method 
    Parameters:
    -----------------------------------------------------------
    * x: list of float data

    * `ax` [`Axes`, optional]:
        The matplotlib axes on which to draw the plot, or `None` to create
        a new one.

    * `confidence` [float, optional]:
        The confidence for validating the abrupt, if known.

    * `series_name` [string, optional]:
        The series name, if known.

    * `fig_id` [string, optional]:
        The index of the ax, if known.

    Returns
    -------
    * `ax`: [`Axes`]:
        The matplotlib axes.
    """
    ax = kwargs.get('ax', None)
    confidence = kwargs.get('confidence', None)
    series_name = kwargs.get('series_name', None)
    fig_id = kwargs.get('fig_id', None)

    if series_name is None:
        series_name = 'Time series'


    if ax is None:
        ax = plt.gca()

    if confidence is None:
        confidence = 0.95

    UBK,idx = get_UBk(x)
    UFK,idx = get_UFk(x)

    
    # crossing = np.argwhere(np.diff(np.sign(UFK_series - UBK_series))).flatten()
    # f1 = interp1d(UFK_series.index, UFK_series.values)
    # f2 = interp1d(UBK_series.index, UBK_series.values)

    # index_at_crossing = [crossing[i] + (crossing[i+1]-crossing[i])*(f1(crossing[i])-f2(crossing[i]))/(f1(crossing[i])-f2(crossing[i+1])+f2(crossing[i])-f1(crossing[i])) for i in range(len(crossing)-1)]

    # print(UFK_series)

    

    if type(idx[0])!=np.int32 or type(idx[0])!=np.int64:
        dates = [datetime.strptime(date, '%Y-%m-%d') for date in idx]
        idx = [date.year for date in dates]


    UFK_series = pd.Series(UFK)
    UBK_series = pd.Series(UBK)

    diff = UFK_series - UBK_series

    # 找到差值符号变化的位置，即可能的交点
    sign_changes = np.sign(diff).diff().fillna(0) != 0

    # print(np.where(sign_changes)[0])
    # 初始化一个列表来存储交点的索引值
    crossing_indices = []

    # 遍历每个符号改变的点，计算精确的交点位置
    for idx_ in np.where(sign_changes)[0]:
        if idx_ > 0:  # 确保有前一个点可以进行插值计算
            # 前后两点进行线性插值
            x1, x2 = idx_ - 1, idx_
            y1, y2 = diff.iloc[x1], diff.iloc[x2]
            # 线性插值公式计算交点的索引
            # 确保不除以零，避免无穷大的情况
            if y2 != y1:
                crossing_index = x1 + (0 - y1) / (y2 - y1)
                crossing_indices.append(crossing_index)

    # 输出交点索引
    mutation_list = [idx[round(i)] for i in crossing_indices]
    mutation_values = [(UFK[round(i)]+UBK[round(i)])/2 for i in crossing_indices] # idx[round(i)]

    # 输出交点索引
    # print("Estimated crossing indices:", crossing_indices)

    Z_alpha = get_Z_alpha(confidence)
    Z_up = Z_alpha*np.ones(len(UFK))
    Z_low = -Z_alpha*np.ones(len(UFK))

    ax.set_ylabel('UF~UB', )
    ax.plot(idx, UFK, color='b', label='UF({})'.format(series_name),zorder=0)
    ax.plot(idx, UBK, color='fuchsia', label='UB({})'.format(series_name),zorder=0)
    ax.plot(idx, Z_up, '--', color='r',label='confidence',zorder=0)
    ax.plot(idx, Z_low, '--', color='r', label='',zorder=0)
    ax.scatter(mutation_list,mutation_values,color='r',zorder=1)
    ax.set_xlabel('Date (Year)', )

    # if fig_id is None:
    #     ax.set_xlabel('Date (Year)', )
    # else:
    #     ax.set_xlabel('Date (Year)\n('+fig_id+')', )

    ylim = ax.get_ylim()
    ax.set_ylim([ylim[0],ylim[1]+0.1*(ylim[1]-ylim[0])])


    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_pos = xlim[0] + 0.02 * (xlim[1] - xlim[0])
    y_pos = ylim[0] + 0.02 * (ylim[1] - ylim[0])
    if fig_id is not None:
        ax.text(x_pos, y_pos, fig_id, ha='left', va='bottom')

    for mu,mu_v in zip(mutation_list,mutation_values):
        ax.text(mu+0.8,mu_v,'{}'.format(mu),color='r',zorder=2)
    ax.set_title('Mann-Kendall Mutation test')
    ax.legend(ncols=3,loc='upper left',shadow=True,frameon=False)
    
    return ax

