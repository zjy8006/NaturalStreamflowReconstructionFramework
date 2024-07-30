import pandas as pd
import matplotlib.pyplot as plt
import MannKendallTrend as mk


tangnaihai_monthly_flow = pd.read_csv('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/data/tangnaihai_monthly_flow.csv',parse_dates=['date'],index_col='date')
guide_monthly_flow = pd.read_csv('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/data/guide_monthly_flow.csv',parse_dates=['date'],index_col='date')
xunhuan_monthly_flow = pd.read_csv('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/data/xunhua_monthly_flow.csv',parse_dates=['date'],index_col='date')

# select data from "1957-01-01" to "2018-12-01"
tangnaihai_monthly_flow = tangnaihai_monthly_flow.loc['1957-01-01':'2018-12-31']
guide_monthly_flow = guide_monthly_flow.loc['1957-01-01':'2018-12-31']
xunhuan_monthly_flow = xunhuan_monthly_flow.loc['1957-01-01':'2018-12-31']


fig = plt.figure(figsize=(7.48,5.5))
ax11 = fig.add_subplot(3,1,1)
ax12 = fig.add_subplot(3,1,2)
ax13 = fig.add_subplot(3,1,3)

ax11.plot(tangnaihai_monthly_flow.index,tangnaihai_monthly_flow['flow(m^3/s)'],color='tab:blue',label='Tangnaihai')
ax12.plot(guide_monthly_flow.index,guide_monthly_flow['flow(m^3/s)'],color='tab:green',label='Guide')
ax13.plot(xunhuan_monthly_flow.index,xunhuan_monthly_flow['flow(m^3/s)'],color='tab:cyan',label='Xunhuan')

# mk.plot_trend(tangnaihai_monthly_flow,ax=ax11,series_name='Tangnaihai')
# mk.plot_trend(guide_monthly_flow,ax=ax12,series_name='Guide')
# mk.plot_trend(xunhuan_monthly_flow,ax=ax13,series_name='Xunhuan')

ax11.axvline(x=pd.to_datetime('1985-12-01'),color='r',linestyle='--',label='1985-12')
ax12.axvline(x=pd.to_datetime('1985-12-01'),color='r',linestyle='--',label='1985-12')
ax13.axvline(x=pd.to_datetime('1985-12-01'),color='r',linestyle='--',label='1985-12')
ax11.set_ylabel('Flow($m^3/s$)')
ax12.set_ylabel('Flow($m^3/s$)')
ax13.set_ylabel('Flow($m^3/s$)')
ax13.set_xlabel('Date(month)')

for ax, fig_id in zip([ax11,ax12,ax13],['(a)','(b)','(c)']):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_pos = xlim[0] + 0.02 * (xlim[1] - xlim[0])
    y_pos = ylim[1] - 0.12 * (ylim[1] - ylim[0])
    ax.text(x_pos, y_pos, fig_id, ha='left', va='bottom')

ax11.legend(ncols=2,shadow=False,frameon=False)
ax12.legend(ncols=2,shadow=False,frameon=False)
ax13.legend(ncols=2,shadow=False,frameon=False)

plt.subplots_adjust(left=0.10, bottom=0.08, right=0.99,top=0.98, hspace=0.2, wspace=0.2)
plt.savefig('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/figs/monthly_streamflow.eps',format='EPS',dpi=2000,transparent=True,bbox_inches='tight')

plt.show()