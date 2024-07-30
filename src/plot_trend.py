import pandas as pd
import calendar
import MannKendallTrend as MKT
import matplotlib.pyplot as plt

def Get_days_from_month(date):
    year = date.year
    month = date.month
    # year, month, day = map(int, date.split('-'))

    # 获取该月的天数
    days_in_month = calendar.monthrange(year, month)[1]

    return days_in_month

tangnaihai = pd.read_csv('data/tangnaihai_annual_runoff.csv',parse_dates=['date'],index_col='date')
tangnaihai = tangnaihai.loc['1957-01-01':'2018-12-31']


guide = pd.read_csv('data/guide_monthly_natural_naturalized_flow.csv',parse_dates=['date'],index_col='date')
guide['date_'] = guide.index
guide['days'] = guide['date_'].apply(Get_days_from_month)
guide['runoff(10^8m^3)'] = guide['VIF_LSTM']*guide['days']*24*3600/100000000
guide = guide.loc[:,['runoff(10^8m^3)']]
guide = guide.resample('YE').sum()


xunhua = pd.read_csv('data/xunhua_monthly_natural_naturalized_flow.csv',parse_dates=['date'],index_col='date')
xunhua['date_'] = xunhua.index
xunhua['days'] = xunhua['date_'].apply(Get_days_from_month)
xunhua['runoff(10^8m^3)'] = xunhua['VIF_LSTM']*xunhua['days']*24*3600/100000000
xunhua = xunhua.loc[:,['runoff(10^8m^3)']]
xunhua = xunhua.resample('YE').sum()


fig = plt.figure(figsize=(7.48,4.0))
ftsize=10
ax11 = fig.add_subplot(1,1,1)

MKT.plot_trend(tangnaihai,ax=ax11,series_name='Tangnaihai',series_color='royalblue',trend_color='orangered')
MKT.plot_trend(guide,ax=ax11,series_name='Guide',series_color='limegreen',trend_color='cyan')
MKT.plot_trend(xunhua,ax=ax11,series_name='Xunhua',series_color='lightslategray',trend_color='indigo')
ax11.set_ylim([100,500])
ax11.set_title('Mann-Kendall trend test',fontsize=ftsize)
ax11.set_xlabel('Date(year)',fontsize=ftsize)
ax11.set_ylabel('Runoff($10^8m^3$)',fontsize=ftsize)
ax11.legend(loc='upper left', ncol=3,shadow=False,frameon=False,fontsize=ftsize)
plt.subplots_adjust(left=0.09, bottom=0.15, right=0.99,top=0.93, hspace=0.35, wspace=0.2)
plt.savefig('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/figs/natural_runoff_trend.eps',format='EPS',dpi=2000,transparent=True,bbox_inches='tight')

plt.show()