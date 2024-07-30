import pandas as pd
import matplotlib.pyplot as plt

import HomogeneityTest as ht
import OrderedClusterAnalysis as oca
import MannKendallMutation as mdm

tangnaihai_annual_runoff = pd.read_csv('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/data/tangnaihai_annual_runoff.csv',parse_dates=['date'],index_col='date')
guide_annual_runoff = pd.read_csv('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/data/guide_annual_runoff.csv',parse_dates=['date'],index_col='date')
xunhuan_annual_runoff = pd.read_csv('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/data/xunhua_annual_runoff.csv',parse_dates=['date'],index_col='date')

# select data from "1957-01-01" to "2018-12-01"
tangnaihai_annual_runoff = tangnaihai_annual_runoff.loc['1957-01-01':'2018-12-31']
guide_annual_runoff = guide_annual_runoff.loc['1957-01-01':'2018-12-31']
xunhuan_annual_runoff = xunhuan_annual_runoff.loc['1957-01-01':'2018-12-31']


fig = plt.figure(figsize=(7.48,6))
ax11 = fig.add_subplot(3,2,1)
ax12 = fig.add_subplot(3,2,2)

ax21 = fig.add_subplot(3,2,3)
ax22 = fig.add_subplot(3,2,4)

ax31 = fig.add_subplot(3,2,5)
ax32 = fig.add_subplot(3,2,6)



# ax31 = fig.add_subplot(3,3,7)
# ax32 = fig.add_subplot(3,3,8)
# ax33 = fig.add_subplot(3,3,9)

# # ht.plot_pettitt(tangnaihai_annual_runoff,ax=ax11,series_name='Tangnaihai')
ht.plot_pettitt(guide_annual_runoff,ax=ax11,series_name='Guide',fig_id='(a)')
oca.plot_abrupt(guide_annual_runoff,ax=ax21,series_name='Guide',fig_id='(b)')
mdm.plot_abrupt(guide_annual_runoff,ax=ax31,series_name='Guide',fig_id='(c)')

ht.plot_pettitt(xunhuan_annual_runoff,ax=ax12,series_name='Xunhuan',fig_id='(d)')
oca.plot_abrupt(xunhuan_annual_runoff,ax=ax22,series_name='Xunhuan',fig_id='(e)')
mdm.plot_abrupt(xunhuan_annual_runoff,ax=ax32,series_name='Xunhuan',fig_id='(f)')

# ax11.axvline(x='1985',color='r',linestyle='--')

ax11.set_xlabel('')
ax12.set_xlabel('')
ax21.set_xlabel('')
ax22.set_xlabel('')
# ax31.set_xlabel('')
# ax32.set_xlabel('')

# ax31.legend(loc='lower left', ncol=3,shadow=False,frameon=False)
# ax32.legend(loc='lower left', ncol=3,shadow=False,frameon=False)

plt.subplots_adjust(left=0.07, bottom=0.06, right=0.99,top=0.96, hspace=0.35, wspace=0.2)
plt.savefig('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/figs/mutation_detection.eps',format='EPS',dpi=2000,transparent=True,bbox_inches='tight')

plt.show()