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
ax11 = fig.add_subplot(3,3,1)
ax12 = fig.add_subplot(3,3,2)
ax13 = fig.add_subplot(3,3,3)

ax21 = fig.add_subplot(3,3,4)
ax22 = fig.add_subplot(3,3,5)
ax23 = fig.add_subplot(3,3,6)

ax31 = fig.add_subplot(3,3,7)
ax32 = fig.add_subplot(3,3,8)
ax33 = fig.add_subplot(3,3,9)



# ax31 = fig.add_subplot(3,3,7)
# ax32 = fig.add_subplot(3,3,8)
# ax33 = fig.add_subplot(3,3,9)

ht.plot_pettitt(tangnaihai_annual_runoff,ax=ax11,series_name='Tangnaihai')
oca.plot_abrupt(tangnaihai_annual_runoff,ax=ax21,series_name='Tangnaihai')
mdm.plot_abrupt(tangnaihai_annual_runoff,ax=ax31,series_name='Tangnaihai')

ht.plot_pettitt(guide_annual_runoff,ax=ax12,series_name='Guide',fig_id='(a)')
oca.plot_abrupt(guide_annual_runoff,ax=ax22,series_name='Guide',fig_id='(b)')
mdm.plot_abrupt(guide_annual_runoff,ax=ax32,series_name='Guide',fig_id='(c)')

ht.plot_pettitt(xunhuan_annual_runoff,ax=ax13,series_name='Xunhuan',fig_id='(d)')
oca.plot_abrupt(xunhuan_annual_runoff,ax=ax23,series_name='Xunhuan',fig_id='(e)')
mdm.plot_abrupt(xunhuan_annual_runoff,ax=ax33,series_name='Xunhuan',fig_id='(f)')

ax11_ylim = ax11.get_ylim()
ax12_ylim = ax12.get_ylim()
ax13_ylim = ax13.get_ylim()

# get the minimum value of ylim[0] from all three subplots
ax1_ylim_min = min(ax11_ylim[0],ax12_ylim[0],ax13_ylim[0])
ax1_ylim_max = max(ax11_ylim[1],ax12_ylim[1],ax13_ylim[1])

ax21_ylim = ax21.get_ylim()
ax22_ylim = ax22.get_ylim()
ax23_ylim = ax23.get_ylim()

ax2_ylim_min = min(ax21_ylim[0],ax22_ylim[0],ax23_ylim[0])
ax2_ylim_max = max(ax21_ylim[1],ax22_ylim[1],ax23_ylim[1])

print(ax2_ylim_min,ax2_ylim_max)

ax31_ylim = ax31.get_ylim()
ax32_ylim = ax32.get_ylim()
ax33_ylim = ax33.get_ylim()


ax3_ylim_min = min(ax31_ylim[0],ax32_ylim[0],ax33_ylim[0])
ax3_ylim_max = max(ax31_ylim[1],ax32_ylim[1],ax33_ylim[1])

ax11.set_ylim([ax1_ylim_min,ax1_ylim_max])
ax12.set_ylim([ax1_ylim_min,ax1_ylim_max])
ax13.set_ylim([ax1_ylim_min,ax1_ylim_max])

ax21.set_ylim([ax2_ylim_min,ax2_ylim_max])
ax22.set_ylim([ax2_ylim_min,ax2_ylim_max])
ax23.set_ylim([ax2_ylim_min,ax2_ylim_max])

ax31.set_ylim([ax3_ylim_min,ax3_ylim_max])
ax32.set_ylim([ax3_ylim_min,ax3_ylim_max])
ax33.set_ylim([ax3_ylim_min,ax3_ylim_max])


# ax11.axvline(x='1985',color='r',linestyle='--')

ax11.set_xlabel('')
ax12.set_xlabel('')
ax13.set_xlabel('')
ax21.set_xlabel('')
ax22.set_xlabel('')
ax23.set_xlabel('')

ax12.set_ylabel('')
ax13.set_ylabel('')
ax22.set_ylabel('')
ax23.set_ylabel('')
ax32.set_ylabel('')
ax33.set_ylabel('')

ax12.set_yticks([])
ax13.set_yticks([])
ax22.set_yticks([])
ax23.set_yticks([])
ax32.set_yticks([])
ax33.set_yticks([])

# ax31.set_xlabel('')
# ax32.set_xlabel('')

# ax31.legend(loc='lower left', ncol=3,shadow=False,frameon=False)
# ax32.legend(loc='lower left', ncol=3,shadow=False,frameon=False)

plt.subplots_adjust(left=0.07, bottom=0.06, right=0.99,top=0.96, hspace=0.3, wspace=0.05)
plt.savefig('D:/ResearchSpace/NaturalStreamflowReconstructionFramework/figs/mutation_detection.eps',format='EPS',dpi=2000,transparent=True,bbox_inches='tight')

plt.show()