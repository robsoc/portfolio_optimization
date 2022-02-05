"""
The below script contains the code used for assessment task i.e. calling portfolio optimization
classess, performance evaluation and plots.

author: Robert Soczewica
date: 15.06.2021
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dataframe_image as dfi
from PortfolioOptimizerSPDR import EquityBond, EqualWeight, MinVar, ERC, TargetReturn

plt.rcParams["figure.figsize"] = (12,6)
plt.style.use('fivethirtyeight')


def get_data(task_num: int) -> pd.DataFrame:
    """
    Function which obtains the data from provided excel file.

    Parameters
    ----------
    task_num : int
        Number of the task

    Returns
    -------
    pd.DataFrame
        Data for the task
    """    
    sheet_name = ' '.join(['Tab', str(task_num)])
    df = pd.read_excel('New Hire_DataFile.xlsx', sheet_name=sheet_name, index_col=0, header=1, engine='openpyxl')
    
    return df


###################
#                 #
#     TASK 1      #
#                 #
###################

data = get_data(task_num=1)
window = 12
bdata = data['World Equities Net Total Return USD Index'].iloc[window:]

# Strategy 1 (60% Equities, 40% Bonds)
eb = EquityBond(data=data,
                equity_col='World Equities Net Total Return USD Index',
                bond_col='Global Bonds Total Return Index Value Unhedged USD',
                )

eb.run(window=window)
dfi.export(eb.summary(bdata=bdata), 'eb_summary.png')
eb.weights.set_index(eb.weights.index.date).plot(kind='bar', stacked=True, title="Strategy 1 (60/40 EB) allocation", colormap='tab10').legend(loc='center left', bbox_to_anchor=(1.0, 0.5));
plt.savefig('weight_plot_eb.png', bbox_inches='tight')

# Strategy 2 (Equal Risk Contribution)
erc = ERC(data=data)
erc.run(window=window)
dfi.export(erc.summary(bdata=bdata), 'erc_summary.png')
erc.weights.set_index(erc.weights.index.date).plot(kind='bar', stacked=True, title="Strategy 2 (ERC) allocation", colormap='tab10').legend(loc='center left', bbox_to_anchor=(1.0, 0.5));
plt.savefig('weight_plot_erc.png', bbox_inches='tight')

# Strategy 3 (Minimum Variance Portfolio)
mvp = MinVar(data=data)
mvp.run(window=window)
dfi.export(mvp.summary(bdata=bdata), 'mvp_summary.png')
mvp.weights.set_index(mvp.weights.index.date).plot(kind='bar', stacked=True, title="Strategy 3 (MVP) allocation", colormap='tab10').legend(loc='center left', bbox_to_anchor=(1.0, 0.5));
plt.savefig('weight_plot_mvp.png', bbox_inches='tight')

# Strategy 4 (Equal Weight Portfolio)
ewp = EqualWeight(data=data)
ewp.run(window=window)
dfi.export(ewp.summary(bdata=bdata), 'ewp_summary.png')
ewp.weights.set_index(ewp.weights.index.date).plot(kind='bar', stacked=True, title="Strategy 4 (EWP) allocation", colormap='tab10').legend(loc='center left', bbox_to_anchor=(1.0, 0.5));
plt.savefig('weight_plot_ewp.png', bbox_inches='tight')

# Plot cumulative returns
plt.plot(eb.cum_ret(eb.pret, total=False), label='Strategy 1 (60/40 EB)')
plt.plot(erc.cum_ret(erc.pret, total=False), label='Strategy 2 (ERC)')
plt.plot(mvp.cum_ret(mvp.pret, total=False), label='Strategy 3 (MVP)')
plt.plot(ewp.cum_ret(ewp.pret, total=False), label='Strategy 4 (EWP)')
plt.legend()
plt.title("Cumulative return of portfolios")
plt.xlabel("Date")
plt.ylabel("Cumulative return")
plt.savefig('cumret_plot_task1.png', bbox_inches='tight')


###################
#                 #
#     TASK 2      #
#                 #
###################

data = get_data(task_num=2)
window = 12
bdata = get_data(task_num=1)['US Large Cap Equities Net Total Return Index'].iloc[window:]

# Strategy 1 (5% Target Return Strategy)
trp = TargetReturn(data=data, target=0.05)
trp.run(window=window)
trp.summary(bdata=bdata)
dfi.export(trp.summary(bdata=bdata), 'trp_summary.png')
trp.weights.set_index(trp.weights.index.date).plot(kind='bar', stacked=True, title="Strategy 1 (5% Target Return) allocation", colormap='turbo').legend(loc='center left', bbox_to_anchor=(1.0, 0.5));
plt.savefig('weight_plot_trp.png', bbox_inches='tight')

# Strategy 2 (Equal Weight Portfolio)
ewp2 = EqualWeight(data=data)
ewp2.run(window=window)
dfi.export(ewp2.summary(bdata=bdata), 'ewp2_summary.png')
ewp2.weights.set_index(ewp2.weights.index.date).plot(kind='bar', stacked=True, title="Strategy 2 (EWP) allocation", colormap='turbo').legend(loc='center left', bbox_to_anchor=(1.0, 0.5));
plt.savefig('weight_plot_ewp2.png', bbox_inches='tight')

# Plot cumulative returns
plt.plot(trp.cum_ret(trp.pret, total=False), label='Strategy 1 (5% Target Return)')
plt.plot(ewp2.cum_ret(ewp2.pret, total=False), label='Strategy 2 (EWP)')
plt.legend()
plt.title("Cumulative return of portfolios")
plt.xlabel("Date")
plt.ylabel("Cumulative return")
plt.savefig('cumret_plot_task2.png', bbox_inches='tight')
