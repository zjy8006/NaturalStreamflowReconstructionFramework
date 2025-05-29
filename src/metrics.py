import numpy as np
import pandas as pd
import math

import numpy as np

def TCS(y_true, y_pred):
    """
    Compute Temporal Consistency Score
    args:
        * `y_true`: the observed records
        * `y_pred`: the predictions
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    diff_true = np.diff(y_true)
    diff_pred = np.diff(y_pred)
    
    product = diff_true * diff_pred
    
    score = np.sum(np.sign(product))
    
    return score / (len(y_true) - 1)

    

def NSE(y_true,y_pred):
    """
    Compute Nash-Sutcliffe efficiency
    args:
        * `y_true`:the observed records
        * `y_pred`:the predictions
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)

def NNSE(y_true,y_pred):
    """
    Compute normalized Nash-Sutcliffe efficiency
    args:
        * `y_true`:the observed records
        * `y_pred`:the predictions
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return 1.0/(2-NSE(y_true,y_pred))

def RMSE(y_true,y_pred):
    """
    Compute root mean square error
    args:
        * `y_true`:the observed records
        * `y_pred`:the predictions
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred)**2))

def NRMSE(y_true,y_pred):
    """
    Compute normalized root mean square error
    args:
        * `y_true`:the observed records
        * `y_pred`:the predictions
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred)**2)) / np.std(y_true)

def ZNRMSE(y_true,y_pred):
    """
    Compute normalized root mean square error
    args:
        * `y_true`:the observed records
        * `y_pred`:the predictions
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)


    y_true_max = np.max(y_true)
    y_pred_max = np.max(y_pred)
    y_max = np.max([y_true_max,y_pred_max])
    y_true_min = np.min(y_true)
    y_pred_min = np.min(y_pred)
    y_min = np.min([y_true_min,y_pred_min])

    return np.sqrt(np.mean((y_true - y_pred)**2)) / (y_max-y_min)

def MAE(y_true,y_pred):
    """
    Compute mean absolute error
    args:
        * `y_true`:the observed records
        * `y_pred`:the predictions
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))



def MAPE(y_true,y_pred):
    """
    Compute mean absolute percentage error
    args:
        * `y_true`:the observed records
        * `y_pred`:the predictions
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def PPTS(y_true,y_pred,gamma=5,norm=True):
    """ 
    Compute peak percentage threshold statistic
    args:
        * `y_true`:the observed records
        * `y_pred`:the predictions
        * `gamma`:lower value percentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true_max = np.max(y_true)
    y_true_min = np.min(y_true)
    y_pred_max = np.max(y_pred)
    y_pred_min = np.min(y_pred)

    max_val = np.max([y_true_max,y_pred_max])
    min_val = np.min([y_true_min,y_pred_min])
    
    err_range = max_val - min_val

    # y_true
    r = pd.DataFrame(y_true,columns=['r'])
    # print('original time series:\n{}'.format(r))
    # predictions
    p = pd.DataFrame(y_pred,columns=['p'])
    # print('predicted time series:\n{}'.format(p))
    # The number of samples
    N = r['r'].size
    # print('series size={}'.format(N))
    # The number of top data
    G = round((gamma/100)*N)
    rp = pd.concat([r,p],axis=1)
    rps=rp.sort_values(by=['r'],ascending=False)
    rps_g=rps.iloc[:G]
    y_true = (rps_g['r']).values
    y_pred = (rps_g['p']).values
    

    
    avg = np.mean(y_true)
    if norm:
        absv=np.abs((y_true-y_pred))/np.abs(err_range)
        sumabsv=np.sum(absv)
        ppts = sumabsv/G
    else:
        abss=np.abs((y_true-y_pred)/y_true*100)
        # print('abs error={}'.format(abss))
        sums = np.sum(abss)
        # print('sum of abs error={}'.format(abss))
        ppts = sums*(1/((gamma/100)*N))
        # print('ppts('+str(gamma)+'%)={}'.format(ppts))
    return ppts

def LPTS(y_true,y_pred,gamma=5,norm=True):
    """ 
    Compute low percentage threshold statistic
    args:
        * `y_true`:the observed records
        * `y_pred`:the predictions
        * `gamma`:lower value percentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true_max = np.max(y_true)
    y_true_min = np.min(y_true)
    y_pred_max = np.max(y_pred)
    y_pred_min = np.min(y_pred)

    max_val = np.max([y_true_max,y_pred_max])
    min_val = np.min([y_true_min,y_pred_min])

    err_range = max_val - min_val


    # y_true
    r = pd.DataFrame(y_true,columns=['r'])
    # print('original time series:\n{}'.format(r))
    # predictions
    p = pd.DataFrame(y_pred,columns=['p'])
    # print('predicted time series:\n{}'.format(p))
    # The number of samples
    N = r['r'].size
    # print('series size={}'.format(N))
    # The number of top data
    G = round((gamma/100)*N)
    rp = pd.concat([r,p],axis=1)
    rps=rp.sort_values(by=['r'],ascending=True)
    rps_g=rps.iloc[:G]
    y_true = (rps_g['r']).values
    y_pred = (rps_g['p']).values
    avg = np.mean(y_true)

    
    if norm:
        absv=np.abs((y_true-y_pred))/np.abs(err_range)
        sumabsv=np.sum(absv)
        lpts = sumabsv/G

    else:
        abss=np.abs((y_true-y_pred)/y_true*100)
        # print('abs error={}'.format(abss))
        sums = np.sum(abss)
        # print('sum of abs error={}'.format(abss))
        lpts = sums*(1/((gamma/100)*N))
        # print('lpts('+str(gamma)+'%)={}'.format(lpts))
    return lpts
def PPMAE(y_true,y_pred,gamma=5):
    """ 
    Compute peak percentage threshold mean absolute error
    args:
        * `y_true`:the observed records
        * `y_pred`:the predictions
        * `gamma`:lower value percentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # y_true
    r = pd.DataFrame(y_true,columns=['r'])
    # print('original time series:\n{}'.format(r))
    # predictions
    p = pd.DataFrame(y_pred,columns=['p'])
    # print('predicted time series:\n{}'.format(p))
    # The number of samples
    N = r['r'].size
    # print('series size={}'.format(N))
    # The number of top data
    G = round((gamma/100)*N)
    rp = pd.concat([r,p],axis=1)
    rps=rp.sort_values(by=['r'],ascending=False)
    rps_g=rps.iloc[:G]
    y_true = (rps_g['r']).values
    y_pred = (rps_g['p']).values
    return np.mean(np.abs(y_true - y_pred))


def LPMAE(y_true,y_pred,gamma=5):
    """ 
    Compute low percentage threshold mean absolute error
    args:
        * `y_true`:the observed records
        * `y_pred`:the predictions
        * `gamma`:lower value percentage
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)



    # y_true
    r = pd.DataFrame(y_true,columns=['r'])
    # print('original time series:\n{}'.format(r))
    # predictions
    p = pd.DataFrame(y_pred,columns=['p'])
    # print('predicted time series:\n{}'.format(p))
    # The number of samples
    N = r['r'].size
    # print('series size={}'.format(N))
    # The number of top data
    G = round((gamma/100)*N)
    rp = pd.concat([r,p],axis=1)
    rps=rp.sort_values(by=['r'],ascending=True)
    rps_g=rps.iloc[:G]
    y_true = (rps_g['r']).values
    y_pred = (rps_g['p']).values
    avg = np.mean(y_true)

    return np.mean(np.abs(y_true - y_pred))



def R2(y_true,y_pred):
    """
    Compute r2 score
    args:
        * `y_true`:the observed records
        * `y_pred`:the predictions
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true_avg=sum(y_true)/len(y_true)
    y_pred_avg=sum(y_pred)/len(y_pred)
    r2=sum((y_true-y_true_avg)*(y_pred-y_pred_avg))/math.sqrt(sum((y_true-y_true_avg)**2)*sum((y_pred-y_pred_avg)**2))
    return r2

