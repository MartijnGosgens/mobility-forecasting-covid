''''
This script generates all figures in the paper.
One can make a distinction between forecasting 7 or 14 days ahead via the variable num_days.
'''

import matplotlib.colors as mplc
import matplotlib.pyplot as plt
import numpy as np
import math
import datetime as dt
import pandas as pd
import seaborn as sn
import copy
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import matplotlib.dates as mdates
import generate_figures as gf
from sklearn.metrics import mean_squared_error, mean_absolute_error

from matplotlib import ticker
tick_locator = ticker.MaxNLocator(nbins=4)

import geopandas as gpd
import mezuro_preprocessing as mzp

import scipy.stats as stats



months = mdates.MonthLocator()  # every month

horizon = 14
start_date = '01-07-2020'
end_date = '31-12-2020'
date_format = '%d-%m-%Y'
start_date_number = dt.datetime.strptime(start_date,date_format)
end_date_number = dt.datetime.strptime(end_date,date_format)

num_days = 14

dates = []
for t in range(0, (end_date_number - start_date_number).days + 1):
    day_difference = dt.timedelta(days = t)
    Current_date_number = start_date_number + day_difference
    Current_date = Current_date_number.strftime('%d-%m-%Y')
    Current_date_plot_str = Current_date_number.strftime('%Y%m%d')
    
    dates.append(Current_date_number)




Trans_rates = pd.read_csv("results_forecasting/Trans_rates_NB_mob_samples_final_new.csv")
Trans_rates_basic = pd.read_csv('results_forecasting/Trans_rates_NB_mob_final_new.csv')
Trans_rates_basic['date'] = Trans_rates_basic['date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y'))
mask = (Trans_rates_basic['date'] >= start_date_number) & (Trans_rates_basic['date'] <= end_date_number) 
Trans_rates_basic = Trans_rates_basic.loc[mask]

Trans_rates_P_mob = pd.read_csv('results_forecasting/Trans_rates_P_mob_final_new.csv')
Trans_rates_P_mob['date'] = Trans_rates_P_mob['date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y'))
mask = (Trans_rates_P_mob['date'] >= start_date_number) & (Trans_rates_P_mob['date'] <= end_date_number) 
Trans_rates_P_mob = Trans_rates_P_mob.loc[mask]

Trans_rates_P_nomob = pd.read_csv('results_forecasting/Trans_rates_P_nomob_final_new.csv')
Trans_rates_P_nomob['date'] = Trans_rates_P_nomob['date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y'))
mask = (Trans_rates_P_nomob['date'] >= start_date_number) & (Trans_rates_P_nomob['date'] <= end_date_number) 
Trans_rates_P_nomob = Trans_rates_P_nomob.loc[mask]

Trans_rates_NB_mob = pd.read_csv('results_forecasting/Trans_rates_NB_mob_final_new.csv')
Trans_rates_NB_mob['date'] = Trans_rates_NB_mob['date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y'))
mask = (Trans_rates_NB_mob['date'] >= start_date_number) & (Trans_rates_NB_mob['date'] <= end_date_number) 
Trans_rates_NB_mob = Trans_rates_NB_mob.loc[mask]

Trans_rates_NB_nomob = pd.read_csv('results_forecasting/Trans_rates_NB_nomob_final_new.csv')
Trans_rates_NB_nomob['date'] = Trans_rates_NB_nomob['date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y'))
mask = (Trans_rates_NB_nomob['date'] >= start_date_number) & (Trans_rates_NB_nomob['date'] <= end_date_number) 
Trans_rates_NB_nomob = Trans_rates_NB_nomob.loc[mask]

df_quantiles = pd.read_csv('results_forecasting/Trans_rates_NB_mob_int_final_new.csv')
df_quantiles['date'] = df_quantiles['date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y'))
mask = (df_quantiles['date'] >= start_date_number) & (df_quantiles['date'] <= end_date_number) 
df_quantiles = df_quantiles.loc[mask]


#Load result files


Results_7days = pd.read_csv('results_forecasting/Results_7days_01072020_31122020_point_new.csv')
Results_7days['date'] = Results_7days['date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y'))
mask = (Results_7days['date'] >= start_date_number) & (Results_7days['date'] <= end_date_number) 
Results_7days = Results_7days.loc[mask]

Results_7days_nomob = pd.read_csv('results_forecasting/Results_7days_01072020_31122020_point_nomob_new.csv')
Results_7days_nomob['date'] = Results_7days_nomob['date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y'))
mask = (Results_7days_nomob['date'] >= start_date_number) & (Results_7days_nomob['date'] <= end_date_number) 
Results_7days_nomob = Results_7days_nomob.loc[mask]

Results_14days = pd.read_csv('results_forecasting/Results_14days_01072020_31122020_point_new.csv')
Results_14days['date'] = Results_14days['date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y'))
mask = (Results_14days['date'] >= start_date_number) & (Results_14days['date'] <= end_date_number) 
Results_14days = Results_14days.loc[mask]

Results_14days_nomob = pd.read_csv('results_forecasting/Results_14days_01072020_31122020_point_nomob_new.csv')
Results_14days_nomob['date'] = Results_14days_nomob['date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y'))
mask = (Results_14days_nomob['date'] >= start_date_number) & (Results_14days_nomob['date'] <= end_date_number) 
Results_14days_nomob = Results_14days_nomob.loc[mask]

if num_days == 7:
    Results_7days_samples = pd.read_csv('results_forecasting/Results_7days_01072020_30092020_new.csv')
    Results_7days_samples = Results_7days_samples.append(pd.read_csv('results_forecasting/Results_7days_01102020_31122020_new.csv'))
else:
    Results_7days_samples = pd.read_csv('results_forecasting/Results_14days_01072020_30092020_new.csv')
    Results_7days_samples = Results_7days_samples.append(pd.read_csv('results_forecasting/Results_14days_01102020_31122020_new.csv'))


Results_7days_samples['date'] = Results_7days_samples['date'].apply(lambda x: dt.datetime.strptime(x, '%d-%m-%Y'))
mask = (Results_7days_samples['date'] >= start_date_number) & (Results_7days_samples['date'] <= end_date_number) 
Results_7days_samples = Results_7days_samples.loc[mask]



#Load validity results on new tests!
df_newtests = pd.read_csv('results_forecasting/Validation_newtests_init.csv')
df_newtests_sim = pd.read_csv('results_forecasting/Validation_newtests.csv')
df_newtests_sim_lower = pd.read_csv('results_forecasting/Validation_newtests_lower.csv')
df_newtests_sim_upper = pd.read_csv('results_forecasting/Validation_newtests_upper.csv')



MSE_7 = []
MSE_7_nomob = []
MSE_diff = []
MAE_7 = []
MAE_7_nomob = []
MAE_diff = []

RMSE_7 = []
RMSE_7_nomob = []
RMSE_diff = []

WMAPE_7 = []
WMAPE_7_nomob = []
WMAPE_diff = []

tot_lower = []
tot_upper = []
tot_predicted = []
tot_outcome = []

Median_error_7 = []
Median_error_7_nomob = []
Median_error_diff = []

R2_7 = []
R2_7_nomob = []


nr_samples = 100
conf_level = .95


HELP_df_7days_samples_tot = Results_7days_samples.groupby(['date','sample']).agg({'pred':'sum'})

for DATE in dates:
    print(DATE)
    
    if num_days == 7:
        HELP_df_7 = Results_7days.groupby('date').get_group(DATE)
        HELP_df_7_nomob = Results_7days_nomob.groupby('date').get_group(DATE)
    else:
        HELP_df_7 = Results_14days.groupby('date').get_group(DATE)
        HELP_df_7_nomob = Results_14days_nomob.groupby('date').get_group(DATE)
            
    
    HELP_MSE = mean_squared_error(HELP_df_7['pred'],HELP_df_7['outcome'])
    HELP_MSE_nomob = mean_squared_error(HELP_df_7_nomob['pred'],HELP_df_7_nomob['outcome'])
    MSE_7.append(HELP_MSE)
    MSE_7_nomob.append(HELP_MSE_nomob)
    MSE_diff.append(1 - HELP_MSE / HELP_MSE_nomob)
    
    HELP_MAE = mean_absolute_error(HELP_df_7['pred'],HELP_df_7['outcome'])
    HELP_MAE_nomob = mean_absolute_error(HELP_df_7_nomob['pred'],HELP_df_7_nomob['outcome'])
    MAE_7.append(HELP_MAE)
    MAE_7_nomob.append(HELP_MAE_nomob)
    MAE_diff.append(1 - HELP_MAE / HELP_MAE_nomob)
    
    HELP_RMSE = np.sqrt(HELP_MSE)
    HELP_RMSE_nomob = np.sqrt(HELP_MSE_nomob)
    RMSE_7.append(HELP_RMSE)
    RMSE_7_nomob.append(HELP_RMSE_nomob)
    RMSE_diff.append(1 - HELP_RMSE / HELP_RMSE_nomob)   
    
    #Compare relative rankings
    predict_tot = sum(HELP_df_7['pred'])
    real_tot = sum(HELP_df_7['outcome'])
    tot_predicted.append(predict_tot)
    tot_outcome.append(real_tot)
    
    predict_percentages = HELP_df_7['pred'] / predict_tot
    real_percentages = HELP_df_7['outcome'] / real_tot
    
    mask = (Results_7days['date'] == DATE)
    Results_7days.loc[mask,'total_pred'] = predict_tot
    Results_7days.loc[mask,'total_real'] = real_tot
    
    HELP_WMAPE = sum(abs(HELP_df_7['pred']-HELP_df_7['outcome'])) / real_tot
    HELP_WMAPE_nomob = sum(abs(HELP_df_7_nomob['pred']-HELP_df_7_nomob['outcome'])) / real_tot
    WMAPE_7.append(HELP_WMAPE)
    WMAPE_7_nomob.append(HELP_WMAPE_nomob)
    WMAPE_diff.append(1 - HELP_WMAPE / HELP_WMAPE_nomob)
    
    HELP_median = np.median((abs(HELP_df_7['pred']-HELP_df_7['outcome'])/HELP_df_7['outcome']).dropna())
    HELP_median_nomob = np.median((abs(HELP_df_7_nomob['pred']-HELP_df_7_nomob['outcome'])/HELP_df_7_nomob['outcome']).dropna())
    Median_error_7.append(HELP_median)
    Median_error_7_nomob.append(HELP_median_nomob)
    Median_error_diff.append(1 - HELP_median / HELP_median_nomob)
    
    HELP_df_7_samples = Results_7days_samples.groupby('date').get_group(DATE)
    
    
    SS_res = sum(pow(HELP_df_7['pred'] - HELP_df_7['outcome'],2))
    SS_tot = sum(pow(HELP_df_7['outcome'] - HELP_df_7['outcome'].mean(),2))
    R2_7.append(1 - SS_res / SS_tot)
    
    SS_res = sum(pow(HELP_df_7_nomob['pred'] - HELP_df_7_nomob['outcome'],2))
    SS_tot = sum(pow(HELP_df_7_nomob['outcome'] - HELP_df_7_nomob['outcome'].mean(),2))
    R2_7_nomob.append(1 - SS_res / SS_tot)
    #list_pred_tot = []
    
    
    '''
    #Compute for each sample totals!
    for sample in range(0,nr_samples):
        HELP_df_7_sample = HELP_df_7_samples.groupby('sample').get_group(sample)
        predict_tot = sum(HELP_df_7_sample['pred'])
        mask2 = (Results_7days_samples['date'] == DATE) & (Results_7days_samples['sample'] == sample)
        Results_7days_samples.loc[mask2,'total_pred'] = predict_tot
        list_pred_tot.append(predict_tot)
    '''
    
    
    HELP_df_samples = HELP_df_7days_samples_tot.groupby('date').get_group(DATE)
    conf_lower = (1 - conf_level) / 2
    conf_upper = (1 + conf_level) / 2
    quant_tot_lower = np.quantile(HELP_df_samples,conf_lower)
    quant_tot_upper = np.quantile(HELP_df_samples,conf_upper)
    Results_7days.loc[mask,'lower_tot'] = quant_tot_lower
    Results_7days.loc[mask,'upper_tot'] = quant_tot_upper
    tot_lower.append(quant_tot_lower)
    tot_upper.append(quant_tot_upper)
        
'''    
for index, row in Trans_rates_NB_mob.iterrows():
    Current_date_number = row['date']
    
    #Adjust mobility:
    if Current_date_number.weekday() == 5:
        Trans_rates_NB_mob.at[index,'mobility'] *= (.36*6.2 / 31)
        Trans_rates_NB_mob.at[index,'rate_mob'] /= (.36*6.2 / 31)
    else:
        if Current_date_number.weekday() == 6:
            Trans_rates_NB_mob.at[index,'mobility'] *= (.27*6.2 / 31)
            Trans_rates_NB_mob.at[index,'rate_mob'] /= (.27*6.2 / 31)
        else:
            Trans_rates_NB_mob.at[index,'mobility'] *= (1 - (.36+.27)*6.2 / 31)
            Trans_rates_NB_mob.at[index,'rate_mob'] /= (1 - (.36+.27)*6.2 / 31)
'''        
    







df_R_RIVM = pd.read_json('data/COVID-19_reproductiegetal.json').set_index('Date')

ylim_lower = 30
ylim_upper = 13000
setpoint_dates = np.exp(np.log(ylim_lower) + 0.05*(np.log(ylim_upper) - np.log(ylim_lower)))
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates, df_newtests['newtests_pred'], c = 'blue', label = 'Via initialization')
ax.plot(dates, df_newtests['newtests'], c= 'black', label = 'RIVM')
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), setpoint_dates), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), setpoint_dates), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), setpoint_dates), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), setpoint_dates), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), setpoint_dates), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), setpoint_dates), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), setpoint_dates), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
#ax.set_ylabel('Effective reproduction number')
ax.set_title('Daily reported positive tests')
#handles,labels = ax.get_legend_handles_labels()
#handles = [handles[0], handles[2], handles[1]]
#labels = [labels[0], labels[2], labels[1]]
#ax.legend(handles, labels, loc = 'upper right')
ax.set_yscale('log')
ax.legend(loc = 'upper left')
plt.ylim([ylim_lower,ylim_upper])
plt.show()
fig.savefig('fig_final_newtests_new.pdf',format="pdf", bbox_inches="tight")


ylim_lower = 15
ylim_upper = 50000
setpoint_dates = np.exp(np.log(ylim_lower) + 0.05*(np.log(ylim_upper) - np.log(ylim_lower)))
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates, df_newtests_sim['newtests_pred'], c='b',label = 'Via simulation')
ax.fill_between(dates, df_newtests_sim_upper['newtests_pred'],df_newtests_sim_lower['newtests_pred'],alpha=.3)
ax.plot(dates, df_newtests['newtests'], c= 'black', label = 'RIVM')
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), setpoint_dates), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), setpoint_dates), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), setpoint_dates), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), setpoint_dates), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), setpoint_dates), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), setpoint_dates), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), setpoint_dates), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
#ax.set_ylabel('Effective reproduction number')
ax.set_title('Daily reported positive tests')
#handles,labels = ax.get_legend_handles_labels()
#handles = [handles[0], handles[2], handles[1]]
#labels = [labels[0], labels[2], labels[1]]
#ax.legend(handles, labels, loc = 'upper right')
ax.set_yscale('log')
ax.legend(loc = 'upper left')
plt.ylim([ylim_lower,ylim_upper])
plt.show()
fig.savefig('fig_final_newtests_sim_new.pdf',format="pdf", bbox_inches="tight")



    
ylim_lower = 0
ylim_upper = 2.5
setpoint_dates = ylim_lower + 0.05*(ylim_upper - ylim_lower)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates, Trans_rates_basic.loc[7:,'R'], c='b',label = 'This paper')
ax.fill_between(dates, df_quantiles.loc[7:,'upper_R'],df_quantiles.loc[7:,'lower_R'],alpha=.3)
ax.plot(dates,df_R_RIVM.loc[start_date_number:end_date_number,'Rt_avg'], c = 'black', label = 'RIVM')
ax.fill_between(dates, df_R_RIVM.loc[start_date_number:end_date_number,'Rt_up'],df_R_RIVM.loc[start_date_number:end_date_number,'Rt_low'],color = 'black',alpha=.3)
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), setpoint_dates), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), setpoint_dates), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), setpoint_dates), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), setpoint_dates), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), setpoint_dates), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), setpoint_dates), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), setpoint_dates), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
#ax.set_ylabel('Effective reproduction number')
ax.set_title('Effective reproduction number')
#handles,labels = ax.get_legend_handles_labels()
#handles = [handles[0], handles[2], handles[1]]
#labels = [labels[0], labels[2], labels[1]]
#ax.legend(handles, labels, loc = 'upper right')
ax.legend(loc = 'upper right')
plt.ylim([ylim_lower,ylim_upper])
plt.show()
fig.savefig('fig_final_R_new.pdf',format="pdf", bbox_inches="tight")

ylim_lower = 0
ylim_upper = 2.5
setpoint_dates = ylim_lower + 0.05*(ylim_upper - ylim_lower)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates, Trans_rates_basic.loc[7:,'R'].shift(-11), c='b',label = 'This paper (shifted by 11 days)')
ax.fill_between(dates, df_quantiles.loc[7:,'upper_R'].shift(-11),df_quantiles.loc[7:,'lower_R'].shift(-11),alpha=.3)
ax.plot(dates,df_R_RIVM.loc[start_date_number:end_date_number,'Rt_avg'], c = 'black', label = 'RIVM')
ax.fill_between(dates, df_R_RIVM.loc[start_date_number:end_date_number,'Rt_up'],df_R_RIVM.loc[start_date_number:end_date_number,'Rt_low'],color = 'black',alpha=.3)
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), setpoint_dates), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), setpoint_dates), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), setpoint_dates), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), setpoint_dates), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), setpoint_dates), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), setpoint_dates), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), setpoint_dates), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
#ax.set_ylabel('Effective reproduction number')
ax.set_title('Effective reproduction number')
#handles,labels = ax.get_legend_handles_labels()
#handles = [handles[0], handles[2], handles[1]]
#labels = [labels[0], labels[2], labels[1]]
#ax.legend(handles, labels, loc = 'upper right')
ax.legend(loc = 'upper right')
plt.ylim([ylim_lower,ylim_upper])
plt.show()
fig.savefig('fig_final_R_shift_new.pdf',format="pdf", bbox_inches="tight")


ylim_lower = 0
ylim_upper = 1
setpoint_dates = ylim_lower + 0.05*(ylim_upper - ylim_lower)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates, Trans_rates_basic.loc[7:,'frac_loc'], c='b')
ax.fill_between(dates, df_quantiles.loc[7:,'upper_fracloc'],df_quantiles.loc[7:,'lower_fracloc'],alpha=.3)
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), setpoint_dates), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), setpoint_dates), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), setpoint_dates), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), setpoint_dates), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), setpoint_dates), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), setpoint_dates), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), setpoint_dates), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
#ax.set_ylabel('Effective R')
ax.set_title('Fraction of local contacts')
#ax.legend(loc = 'upper right')
plt.ylim([ylim_lower,ylim_upper])
plt.show()
fig.savefig('fig_final_fracloc_new.pdf',format="pdf", bbox_inches="tight")


ylim_lower = 0
ylim_upper = .2
setpoint_dates = ylim_lower + 0.05*(ylim_upper - ylim_lower)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates, Trans_rates_basic.loc[7:,'rate_loc'], c='b')
ax.fill_between(dates, df_quantiles.loc[7:,'upper_loc'],df_quantiles.loc[7:,'lower_loc'],alpha=.3)
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), setpoint_dates), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), setpoint_dates), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), setpoint_dates), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), setpoint_dates), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), setpoint_dates), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), setpoint_dates), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), setpoint_dates), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
#ax.set_ylabel('Effective R')
ax.set_title('Transmission rate - local contacts')
#ax.legend(loc = 'upper right')
plt.ylim([ylim_lower,ylim_upper])
plt.show()
fig.savefig('fig_final_loc_new.pdf',format="pdf", bbox_inches="tight")


ylim_lower = 0
ylim_upper = .55
setpoint_dates = ylim_lower + 0.05*(ylim_upper - ylim_lower)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates, Trans_rates_NB_mob.loc[7:,'rate_mob'], c='b',)
ax.fill_between(dates, df_quantiles.loc[7:,'upper_mob'],df_quantiles.loc[7:,'lower_mob'],alpha=.3)
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), setpoint_dates), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), setpoint_dates), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), setpoint_dates), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), setpoint_dates), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), setpoint_dates), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), setpoint_dates), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), setpoint_dates), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
#ax.set_ylabel('Effective R')
ax.set_title('Transmission rate - contacts due to travelling')
#ax.legend(loc = 'upper right')
plt.ylim([ylim_lower,ylim_upper])
plt.show()
fig.savefig('fig_final_mob_new.pdf',format="pdf", bbox_inches="tight")


ylim_lower = 0
ylim_upper = .3
setpoint_dates = ylim_lower + 0.05*(ylim_upper - ylim_lower)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates, Trans_rates_basic.loc[7:,'prob_times_contacts'], c='b')
ax.fill_between(dates, df_quantiles.loc[7:,'upper_newlyinfected'],df_quantiles.loc[7:,'lower_newlyinfected'],alpha=.3)
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), setpoint_dates), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), setpoint_dates), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), setpoint_dates), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), setpoint_dates), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), setpoint_dates), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), setpoint_dates), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), setpoint_dates), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
#ax.set_ylabel('Effective R')
ax.set_title('Transmission probability times average contact rate')
#ax.legend(loc = 'upper right')
plt.ylim([ylim_lower,ylim_upper])
plt.show()
fig.savefig('fig_final_newlyinfected_new.pdf',format="pdf", bbox_inches="tight")



ylim_lower = 0
ylim_upper = 12000000
setpoint_dates = ylim_lower + 0.05*(ylim_upper - ylim_lower)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates, Trans_rates_NB_mob.loc[7:,'mobility'], c='b')
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), setpoint_dates), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), setpoint_dates), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), setpoint_dates), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), setpoint_dates), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), setpoint_dates), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), setpoint_dates), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), setpoint_dates), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
#ax.set_ylabel('Effective R')
ax.set_title('Total daily mobility')
plt.ylim([ylim_lower,ylim_upper])
plt.show()
fig.savefig('fig_final_mobility_new.pdf',format="pdf", bbox_inches="tight")





ylim_lower = -5
ylim_upper = 1350
setpoint_dates = ylim_lower + 0.05*(ylim_upper - ylim_lower)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates, Trans_rates_P_mob.loc[7:,'GoF'] - Trans_rates_NB_mob.loc[7:,'GoF'], c='b',label = 'Poisson with mobility minus NB with mobility')
ax.plot(dates, Trans_rates_NB_nomob.loc[7:,'GoF'] - Trans_rates_NB_mob.loc[7:,'GoF'], c='black', label= 'NB without mobility minus NB with mobility')
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), setpoint_dates), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), setpoint_dates), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), setpoint_dates), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), setpoint_dates), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), setpoint_dates), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), setpoint_dates), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), setpoint_dates), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
ax.set_ylabel('Difference')
ax.set_title('Difference in AIC between transmission rate models ')
#ax.legend(loc = 'center left',bbox_to_anchor=(0,0.25))
ax.legend(loc = 'upper right')
plt.ylim([ylim_lower,ylim_upper])
plt.show()
fig.savefig('fig_final_AIC_new.pdf',format="pdf", bbox_inches="tight")

ylim_lower = -5
ylim_upper = 15
setpoint_dates = ylim_lower + 0.05*(ylim_upper - ylim_lower)
fig, ax = plt.subplots(figsize=(10,4))
#ax.plot(dates, Trans_rates_P_mob.loc[7:,'GoF'] - Trans_rates_NB_mob.loc[7:,'GoF'], c='b',label = 'Poisson with mobility minus NB with mobility')
ax.plot(dates, Trans_rates_NB_nomob.loc[7:,'GoF'] - Trans_rates_NB_mob.loc[7:,'GoF'], c='blue', label= 'NB without mobility minus NB with mobility')
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), setpoint_dates), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), setpoint_dates), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), setpoint_dates), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), setpoint_dates), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), setpoint_dates), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), setpoint_dates), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), setpoint_dates), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
ax.set_ylabel('Difference')
ax.set_title('Difference in AIC between transmission rate models ')
#ax.legend(loc = 'center left',bbox_to_anchor=(0,0.25))
#ax.legend(loc = 'upper right')
plt.ylim([ylim_lower,ylim_upper])
plt.show()
fig.savefig('fig_final_AIC_zoom_new.pdf',format="pdf", bbox_inches="tight")


ylim_lower = -5
ylim_upper = 250
setpoint_dates = ylim_lower + 0.05*(ylim_upper - ylim_lower)
fig, ax = plt.subplots(figsize=(10,4))
#ax.plot(dates, Trans_rates_P_mob.loc[7:,'GoF'] - Trans_rates_NB_mob.loc[7:,'GoF'], c='b',label = 'Poisson with mobility minus NB with mobility')
ax.plot(dates, Trans_rates_NB_nomob.loc[7:,'GoF'] - Trans_rates_NB_mob.loc[7:,'GoF'], c='black', label= 'NB without mobility minus NB with mobility')
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), setpoint_dates), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), setpoint_dates), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), setpoint_dates), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), setpoint_dates), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), setpoint_dates), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), setpoint_dates), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), setpoint_dates), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
ax.set_ylabel('Difference')
ax.set_title('Difference in AIC between transmission rate models ')
#ax.legend(loc = 'center left',bbox_to_anchor=(0,0.25))
#ax.legend(loc = 'upper right')
plt.ylim([ylim_lower,ylim_upper])
plt.show()
fig.savefig('fig_final_AIC_NB_new.pdf',format="pdf", bbox_inches="tight")





ylim_lower = -1
ylim_upper = 1
setpoint_dates = ylim_lower + 0.05*(ylim_upper - ylim_lower)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates, MSE_diff, label = 'Decrease in MSE',c='b')
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), setpoint_dates), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), setpoint_dates), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), setpoint_dates), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), setpoint_dates), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), setpoint_dates), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), setpoint_dates), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), setpoint_dates), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
ax.set_ylabel('Decrease in error')
ax.set_title('Impact of mobility on prediction quality')
plt.ylim([ylim_lower,ylim_upper])
plt.show()
fig.savefig('fig_final_decrease_error_'+str(num_days)+'_new.pdf',format="pdf", bbox_inches="tight")

ylim_lower = -1.6
ylim_upper = .85
setpoint_dates = ylim_lower + 0.05*(ylim_upper - ylim_lower)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates, RMSE_diff, label = 'Decrease in RMSE',c='b')
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), setpoint_dates), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), setpoint_dates), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), setpoint_dates), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), setpoint_dates), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), setpoint_dates), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), setpoint_dates), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), setpoint_dates), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
ax.set_ylabel('Decrease in RMSE')
ax.set_title('Impact of mobility on prediction quality')
plt.ylim([ylim_lower,ylim_upper])
plt.show()
fig.savefig('fig_final_decrease_RMSE_error_'+str(num_days)+'_new.pdf',format="pdf", bbox_inches="tight")

ylim_lower = -.5
ylim_upper = .75
setpoint_dates = ylim_lower + 0.05*(ylim_upper - ylim_lower)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates, WMAPE_diff, label = 'Decrease in WMAPE',c='b')
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), setpoint_dates), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), setpoint_dates), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), setpoint_dates), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), setpoint_dates), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), setpoint_dates), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), setpoint_dates), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), setpoint_dates), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
ax.set_ylabel('Decrease in WMAPE')
ax.set_title('Impact of mobility on prediction quality')
plt.ylim([ylim_lower,ylim_upper])
plt.show()
fig.savefig('fig_final_decrease_WMAPE_error_'+str(num_days)+'_new.pdf',format="pdf", bbox_inches="tight")

if num_days == 7:
    ylim_lower = 0
    ylim_upper = 300
else:
    ylim_lower = 0
    ylim_upper = 800
setpoint_dates = ylim_lower + 0.05*(ylim_upper - ylim_lower)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates, RMSE_7, label = 'RMSE with mobility',c='b')
ax.plot(dates, RMSE_7_nomob, label = 'RMSE without mobility',c='black')
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), setpoint_dates), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), setpoint_dates), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), setpoint_dates), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), setpoint_dates), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), setpoint_dates), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), setpoint_dates), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), setpoint_dates), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
ax.set_ylabel('RMSE')
ax.set_title('Root mean squared error of forecasts with and without mobility')
ax.legend(loc = 'upper left')
plt.ylim([ylim_lower,ylim_upper])
plt.show()
fig.savefig('fig_final_RMSE_'+str(num_days)+'_new.pdf',format="pdf", bbox_inches="tight")



#Compute historical concentrations takes much time, therefore outcommented!
#gf.compute_historical_concentrations(date = start_date_number, date_until = end_date_number + dt.timedelta(days = 1))
concentrations = pd.read_csv("results/historical_concentration.csv").set_index('date')

ylim_lower = 0
ylim_upper = 1
setpoint_dates = ylim_lower + 0.05*(ylim_upper - ylim_lower)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates, concentrations, c='b')
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), 0.05), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), 0.05), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), 0.05), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), 0.05), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), 0.05), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), 0.05), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), 0.05), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
ax.set_ylabel('Concentration')
ax.set_title('Concentration of infections over municipalities')
plt.ylim([0,1])
plt.show()
fig.savefig('fig_final_concentration_new.pdf',format="pdf", bbox_inches="tight")

if num_days == 7:
    ylim_lower = 100
    ylim_upper = 150000
else:
    ylim_lower = 300
    ylim_upper = 300000
setpoint_dates = np.exp(np.log(ylim_lower) + 0.05*(np.log(ylim_upper) - np.log(ylim_lower)))
#setpoint_dates = ylim_lower + 0.05*(ylim_upper - ylim_lower)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates, tot_predicted, c='b',label = 'Forecast reported cases')
ax.fill_between(dates, tot_upper,tot_lower,alpha=.3)
ax.plot(dates, tot_outcome, c='black', label = 'Observed reported cases')
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), setpoint_dates), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), setpoint_dates), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), setpoint_dates), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), setpoint_dates), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), setpoint_dates), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), setpoint_dates), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), setpoint_dates), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
#ax.set_ylabel('Difference')
ax.set_title('Number of forecast reported cases')
#ax.legend(loc = 'center left',bbox_to_anchor=(0,0.25))
ax.legend(loc = 'upper left')
ax.set_yscale('log')
plt.ylim([ylim_lower,ylim_upper])
plt.show()
fig.savefig('fig_final_tot_'+str(num_days)+'_new.pdf',format="pdf", bbox_inches="tight")


fraction_corr = []
fraction_corr_nomob = []
for date_nr in dates:
    date = date_nr.strftime(date_format)
    Current_date_plot_str = date
    #date = '02-10-2020'
    date_nr = datetime.strptime(date,date_format)
    #plot_percentage_result(ax,Results_7days,'Result test',test_date_nr,vmax = .1)
        #df = load_initialization(init_file)
        # Compute fraction infections / fraction inhabitants
        
        
    current_date_end_nr = date_nr + dt.timedelta(days = num_days)
    Current_end_str = current_date_end_nr.strftime(date_format)
        
        
        
        
    HELP_df = Results_7days.groupby('date').get_group(date_nr)
    
    HELP_df_nomob = Results_7days_nomob.groupby('date').get_group(date_nr)
    
    '''
    fig= plt.figure()
    ax = fig.add_axes([0.15, 0.15, .7, .7]) # main axes
    ax.scatter(HELP_df['outcome'], HELP_df['pred'])   
    #for i, txt in enumerate(Breda_wijken):
    #    ax.annotate(txt, (diff_real[i], diff[i]), size = 8)
    ax.plot(np.arange(0, 2500),np.arange(0,2500),'g')
    #ax.plot(np.arange(-1000,1000),[0]*2000,'g')
    #ax.plot([0]*2000,np.arange(-1000,1000),'g')
    ax.set_xlabel('Reported infections')
    ax.set_ylabel('Predicted reported infections')
    ax.set_title('Predicted vs. realized reported infetions during last 7 days \n Prediction made on ' + date + ' for '+ Current_end_str)
    #ax.set_xticks([0,100,500,1000,2000])
    #ax.set_xticklabels(['07','08','09','10','11'])
    #ax.set_yticks([0,100,500,1000,2000])
    #ax.annotate(xy=(x[7], 0.75), s="Historical (RIVM)", c='b')
    #ax.legend()
    max_value = max(max(HELP_df['outcome']),max(HELP_df['pred']))
    max_limit = math.pow(10,math.ceil(math.log10(max_value)))
    ax.set_xlim([1,max_limit])
    ax.set_ylim([1,max_limit])
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()
    fig.savefig('fig_scatter_7_' + str(Current_date_plot_str) + '.png')
    '''
    
    
    pred_tot = sum(HELP_df['pred'])
    real_tot = sum(HELP_df['outcome'])
    HELP_df['pred_fraction'] = HELP_df['pred'] / pred_tot
    HELP_df['real_fraction'] = HELP_df['outcome'] / real_tot
    
    
    
    pred_tot_nomob = sum(HELP_df_nomob['pred'])
    real_tot_nomob = sum(HELP_df_nomob['outcome'])
    HELP_df_nomob['pred_fraction'] = HELP_df_nomob['pred'] / pred_tot_nomob
    HELP_df_nomob['real_fraction'] = HELP_df_nomob['outcome'] / real_tot_nomob
    
    # add geometry
    HELP_df = gpd.GeoDataFrame(HELP_df.merge(mzp.gemeente_shapes.reset_index(), left_on='name', right_on ='name'))
    HELP_df_nomob = gpd.GeoDataFrame(HELP_df_nomob.merge(mzp.gemeente_shapes.reset_index(), left_on='name', right_on ='name'))
 
    # Plot
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize=(7,5), sharey = True)
    ax1.axis('off')
    ax1.set_title('%s' % 'Forecast made on ' + date + '\n for until ' + Current_end_str,fontsize=10)
    #plt.figtext(0.25,0.15,'Forecast on ' + date + ' for ' + Current_end_str)
    HELP_df.plot(ax= ax1, edgecolor = 'black', linewidth = .1, column = 'pred_fraction',cmap='OrRd',legend=False, norm=mplc.LogNorm(vmin=max(.000001,min(HELP_df['pred_fraction'])), vmax=max(max(HELP_df['pred_fraction']),.1)))
    
    #fig, ax = plt.subplots(figsize=(10,4))
    ax2.axis('off')
    ax2.set_title('%s' % 'Observation from ' + date + '\n until ' + Current_end_str,fontsize=10)
    #plt.figtext(0.75,0.15,'Outcome on ' + Current_end_str)
    HELP_df.plot(ax = ax2, edgecolor='black', linewidth = .1, column = 'real_fraction',cmap='OrRd',legend=False,  norm=mplc.LogNorm(vmin=max(.000001,min(HELP_df['pred_fraction'])), vmax=max(max(HELP_df['pred_fraction']) , .1)))
    
    sm = plt.cm.ScalarMappable(cmap='OrRd',  norm=mplc.LogNorm(vmin=max(.000001,min(HELP_df['pred_fraction'])), vmax=max(max(HELP_df['pred_fraction']) , .1)))
    sm._A = []
    cbar_ax = fig.add_axes([0.15, 0.15, 0.7, 0.05])
    cb = fig.colorbar(sm, cax=cbar_ax,orientation='horizontal')
    cb.set_label(label="Fraction of all reported infections")

    
    fig.savefig('figures/fraction_map_'+str(num_days)+'/fig_fraction_map_'+str(num_days)+'_'+str(Current_date_plot_str) + '_new.pdf',format="pdf", bbox_inches="tight")
    

    '''
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.05])
    cb = fig.colorbar(sm, cax=cbar_ax,orientation='horizontal')
    cb.set_label(label="Active infections density / population density", size=18)
    cb.ax.tick_params(labelsize=18)
    '''
    
    
    
     #Alternative kaart-plot:
         
         
    fig, (ax1,ax2) = plt.subplots(nrows = 1, ncols = 2, figsize=(7,5), sharey = True)
    ax1.axis('off')
    ax1.set_title('%s' % 'Error in forecast made on \n' + date + ' for until ' + Current_end_str,fontsize=10)
    #plt.figtext(0.25,0.15,'Forecast on ' + date + ' for ' + Current_end_str)
    HELP_df['diff_per_100000'] = (HELP_df['outcome']-HELP_df['pred']) / HELP_df['AANT_INW'] * 100000
    
    
    min_value = min(HELP_df['diff_per_100000'])
    max_value = max(HELP_df['diff_per_100000'])
    max_limit = math.pow(2,math.ceil(math.log2(max_value)))
    #max_limit = 2*math.ceil(max_value / 2)
    
    if min_value >= 0:
        min_limit = 0
    else:
        min_limit = -1* math.pow(2,math.ceil(math.log2(-1*min_value)))
      
    divnorm=mplc.TwoSlopeNorm(vmin=min_limit, vcenter=0, vmax=max_limit)
    HELP_df.plot(ax= ax1, edgecolor = 'black', linewidth = .1, column = 'diff_per_100000',cmap='RdBu',norm=divnorm,legend=False, legend_kwds ={'orientation':'horizontal','label':'Absolute error per 100,000','cax':fig.add_axes([0.15, 0.175, 0.3, 0.025])})

    ax2.axis('off')
    ax2.set_title('%s' % 'Total reported cases between \n' + date + ' and ' + Current_end_str,fontsize=10)
    #plt.figtext(0.75,0.15,'Outcome on ' + Current_end_str)
    #HELP_df['diff_p'] = HELP_df['diff'] / HELP_df['AANT_INW'] * 100000
    HELP_df['outcome_per_100000'] = HELP_df['outcome'] / HELP_df['AANT_INW'] * 100000
    
    
    max_value = max(HELP_df['outcome_per_100000'])
    max_limit = math.pow(2,math.ceil(math.log2(max_value)))
    #max_limit = max(max_limit,2*math.ceil(max_value / 2))
    
    HELP_df.plot(ax= ax2, edgecolor = 'black', linewidth = .1, column = 'outcome_per_100000',cmap='OrRd', norm=plt.Normalize(vmin=0, vmax=max_limit),legend=False,legend_kwds ={'orientation':'horizontal','label':'Total per 100,000','cax':fig.add_axes([0.575, 0.175, 0.3, 0.025])} )
    
    
    
    


    
    sm = plt.cm.ScalarMappable(cmap='RdBu',  norm = divnorm)
    sm._A = []
    cbar_ax = fig.add_axes([0.15, 0.175, 0.3, 0.025])
    cb = fig.colorbar(sm, cax=cbar_ax,orientation='horizontal')
    cb.set_label(label="")
    cb.locator = tick_locator
    cb.update_ticks()
    cb.set_label(label="Error per 100,000")
    
    sm = plt.cm.ScalarMappable(cmap='OrRd',  norm=plt.Normalize(vmin=0, vmax=max_limit))
    sm._A = []
    cbar_ax = fig.add_axes([0.575, 0.175, 0.3, 0.025])
    cb = fig.colorbar(sm, cax=cbar_ax,orientation='horizontal')
    cb.set_label(label="")
    cb.set_label(label="Total per 100,000")
    
    fig.savefig('figures/diff_per_map_'+str(num_days)+'/fig_diff_per_map_'+str(num_days)+'_'+str(Current_date_plot_str) + '_new.jpg',dpi = 400, bbox_inches="tight")
  
    





    fig, (ax1,ax2) = plt.subplots(nrows = 1, ncols = 2, figsize=(7,5), sharey = True)
    ax1.axis('off')
    ax1.set_title('%s' % 'Error in forecast made on \n' + date + ' for until ' + Current_end_str,fontsize=10)
    #plt.figtext(0.25,0.15,'Forecast on ' + date + ' for ' + Current_end_str)
    HELP_df['diff_no_abs'] = HELP_df['outcome'] - HELP_df['pred'] 
    
    min_value = min(HELP_df['diff_no_abs'])
    max_value = max(HELP_df['diff_no_abs'])
    max_limit = math.pow(2,math.ceil(math.log2(max_value)))
    #max_limit = 2*math.ceil(max_value / 2)
    
    if min_value >= 0:
        min_limit = 0
    else:
        min_limit = -1* math.pow(2,math.ceil(math.log2(-1*min_value)))
    
    divnorm=mplc.TwoSlopeNorm(vmin=min_limit, vcenter=0, vmax=max_limit)
    
    HELP_df.plot(ax= ax1, edgecolor = 'black', linewidth = .1, column = 'diff_no_abs',cmap='RdBu',norm = divnorm,legend=False, legend_kwds ={'orientation':'horizontal','label':'Absolute error'})

    ax2.axis('off')
    ax2.set_title('%s' % 'Total reported cases between \n' + date + ' and ' + Current_end_str,fontsize=10)
    #plt.figtext(0.75,0.15,'Outcome on ' + Current_end_str)
    #HELP_df['diff_p'] = HELP_df['diff'] / HELP_df['AANT_INW'] * 100000
    #HELP_df['outcome_per_100000'] = HELP_df['outcome'] / HELP_df['AANT_INW'] * 100000
    
    
    max_value = max(HELP_df['outcome'])
    max_limit = math.pow(2,math.ceil(math.log2(max_value)))
    #max_limit = max(max_limit,2*math.ceil(max_value / 2))
    
    HELP_df.plot(ax= ax2, edgecolor = 'black', linewidth = .1, column = 'outcome',cmap='OrRd', norm=plt.Normalize(vmin=0, vmax=max_limit),legend=False,legend_kwds ={'orientation':'horizontal','label':'Total'} )
    

    
    sm = plt.cm.ScalarMappable(cmap='RdBu',  norm = divnorm)
    sm._A = []
    cbar_ax = fig.add_axes([0.15, 0.175, 0.3, 0.025])
    cb = fig.colorbar(sm, cax=cbar_ax,orientation='horizontal')
    cb.set_label(label="")
    cb.locator = tick_locator
    cb.update_ticks()
    cb.set_label(label="Error")
    
    sm = plt.cm.ScalarMappable(cmap='OrRd',  norm=plt.Normalize(vmin=0, vmax=max_limit))
    sm._A = []
    cbar_ax = fig.add_axes([0.575, 0.175, 0.3, 0.025])
    cb = fig.colorbar(sm, cax=cbar_ax,orientation='horizontal')
    cb.set_label(label="")
    cb.set_label(label="Total")
    




    
    fig.savefig('figures/diff_map_'+str(num_days)+'/fig_diff_map_'+str(num_days)+'_'+str(Current_date_plot_str) + '_new.pdf',format="pdf", bbox_inches="tight")
      
    
    '''
    fig, ax2 = plt.subplots(nrows = 1, ncols = 1, figsize=(3.5,5), sharey = True)
     
    #fig, ax = plt.subplots(figsize=(10,4))
    ax2.axis('off')
    ax2.set_title('%s' % 'Forecast made on ' + date + '\n for until ' + Current_end_str,fontsize=10)
    #plt.figtext(0.75,0.15,'Outcome on ' + Current_end_str)
    #HELP_df['diff_p'] = HELP_df['diff'] / HELP_df['AANT_INW'] * 100000
    
    max_value = max(HELP_df['diff'])
    #max_limit = math.pow(10,math.ceil(math.log10(max_value)))
    max_limit = 2*math.ceil(max_value / 2)
    
    HELP_df.plot(ax= ax2, edgecolor = 'black', linewidth = .1, column = 'diff',cmap='OrRd',legend=False)
    
    sm = plt.cm.ScalarMappable(cmap='OrRd',  norm=plt.Normalize(vmin=0, vmax=max_limit))
    sm._A = []
    cbar_ax = fig.add_axes([0.15, 0.15, 0.7, 0.05])
    cb = fig.colorbar(sm, cax=cbar_ax,orientation='horizontal')
    cb.set_label(label="Absolute forecast error")
    fig.savefig('figures/diff_map_'+str(num_days)+'/fig_diff_map_'+str(num_days)+'_'+str(Current_date_plot_str) + '_new.png')
    '''
    
 
    
 
    
    #fig= plt.figure()
    #ax = fig.add_axes([0.05, 0.05, 1, 1]) # main axes
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(HELP_df['real_fraction'], HELP_df['pred_fraction'])   
    #for i, txt in enumerate(Breda_wijken):
    #    ax.annotate(txt, (diff_real[i], diff[i]), size = 8)
    ax.plot(np.arange(0, 250000),np.arange(0,250000),'g')
    #ax.plot(np.arange(-1000,1000),[0]*2000,'g')
    #ax.plot([0]*2000,np.arange(-1000,1000),'g')
    ax.set_xlabel('Observed fraction of reported infections')
    ax.set_ylabel('Forecast fraction of reported infections')
    ax.set_title('Forecast vs. observed fraction of reported infections during next '+str(num_days)+' days \n Forecast made on ' + date + ' for until '+ Current_end_str)
    #ax.set_xticks([0,100,500,1000,2000])
    #ax.set_xticklabels(['07','08','09','10','11'])
    #ax.set_yticks([0,100,500,1000,2000])
    #ax.annotate(xy=(x[7], 0.75), s="Historical (RIVM)", c='b')
    #ax.legend()
    max_value = max(max(HELP_df['outcome']),max(HELP_df['pred']))
    max_limit = math.pow(10,math.ceil(math.log10(max_value)))
    ax.set_xlim([max(.00001,min(HELP_df['pred_fraction'])),max(max(HELP_df['pred_fraction']),.3)])
    ax.set_ylim([max(.00001,min(HELP_df['pred_fraction'])),max(max(HELP_df['pred_fraction']),.3)])
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()
    fig.savefig('figures/scatter_fraction_'+str(num_days)+'/fig_scatter_fraction_'+str(num_days)+'_' + str(Current_date_plot_str) + '_new.pdf',format="pdf", bbox_inches="tight")
    
    
    
    
    
    
    
    '''
    
    fig= plt.figure()
    ax = fig.add_axes([0.15, 0.15, .7, .7]) # main axes
    ax.scatter(HELP_df['outcome'], HELP_df['pred'])   
    #for i, txt in enumerate(Breda_wijken):
    #    ax.annotate(txt, (diff_real[i], diff[i]), size = 8)
    ax.plot(np.arange(0, 2500),np.arange(0,2500),'g')
    #ax.plot(np.arange(-1000,1000),[0]*2000,'g')
    #ax.plot([0]*2000,np.arange(-1000,1000),'g')
    ax.set_xlabel('Reported infections')
    ax.set_ylabel('Predicted reported infections')
    ax.set_title('Predicted vs. realized reported infetions during next 7 days \n Prediction made on ' + date + ' for '+ Current_end_str)
    #ax.set_xticks([0,100,500,1000,2000])
    #ax.set_xticklabels(['07','08','09','10','11'])
    #ax.set_yticks([0,100,500,1000,2000])
    #ax.annotate(xy=(x[7], 0.75), s="Historical (RIVM)", c='b')
    #ax.legend()
    max_value = max(max(HELP_df['outcome']),max(HELP_df['pred']))
    max_limit = math.pow(10,math.ceil(math.log10(max_value)))
    ax.set_xlim([1,max_limit])
    ax.set_ylim([1,max_limit])
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.show()
    fig.savefig('fig_scatter_7_' + str(Current_date_plot_str) + '.png')
    '''
    
    
    
    
    
    #fig= plt.figure()
    #ax = fig.add_axes([0.05, 0.05, 1, 1]) # main axes
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(100000*HELP_df['outcome']/HELP_df['AANT_INW'],100000* HELP_df['pred']/HELP_df['AANT_INW'])   
    #for i, txt in enumerate(Breda_wijken):
    #    ax.annotate(txt, (diff_real[i], diff[i]), size = 8)
    ax.plot(np.arange(0, 250000),np.arange(0,250000),'g')
    #ax.plot(np.arange(-1000,1000),[0]*2000,'g')
    #ax.plot([0]*2000,np.arange(-1000,1000),'g')
    ax.set_xlabel('Observed reported infections per 100,000')
    ax.set_ylabel('Forecast reported infections per 100,000')
    ax.set_title('Forecast vs. observed reported infections during next '+str(num_days)+' days \n Forecast made on ' + date + ' for until '+ Current_end_str)
    #ax.set_xticks([0,100,500,1000,2000])
    #ax.set_xticklabels(['07','08','09','10','11'])
    #ax.set_yticks([0,100,500,1000,2000])
    #ax.annotate(xy=(x[7], 0.75), s="Historical (RIVM)", c='b')
    #ax.legend()
    max_value = 100000*max(max(HELP_df['outcome']/HELP_df['AANT_INW']),max(HELP_df['pred']/HELP_df['AANT_INW']))
    #max_limit = math.pow(10,math.ceil(math.log10(max_value)))
    max_limit = 100*math.ceil(max_value / 100)
    ax.set_xlim([0,max_limit])
    ax.set_ylim([0,max_limit])
    plt.show()
    fig.savefig('figures/scatter_'+str(num_days)+'/fig_scatter_'+str(num_days)+'_' + str(Current_date_plot_str) + '_new.pdf',format="pdf", bbox_inches="tight")
        
    
    #Compute order correlaction
    rank_list_pred = HELP_df['pred_fraction'].rank()
    rank_list_real = HELP_df['real_fraction'].rank()
    [corr,pvalue] = stats.pearsonr(rank_list_pred,rank_list_real)
    [corr,pvalue] = stats.spearmanr(HELP_df['pred'] / HELP_df['AANT_INW'] *100000, HELP_df['outcome'] / HELP_df['AANT_INW'] *100000)

    fraction_corr.append(corr)
    
    #For no_mob:
    rank_list_pred = HELP_df_nomob['pred_fraction'].rank()
    rank_list_real = HELP_df_nomob['real_fraction'].rank()
    [corr,pvalue] = stats.pearsonr(rank_list_pred,rank_list_real)
    [corr,pvalue] = stats.spearmanr(HELP_df_nomob['pred'] / HELP_df_nomob['AANT_INW'] *100000, HELP_df_nomob['outcome'] / HELP_df_nomob['AANT_INW'] *100000)
    fraction_corr_nomob.append(corr)
 



ylim_lower = -1
ylim_upper = 1
setpoint_dates = ylim_lower + 0.05*(ylim_upper - ylim_lower)
fig, ax = plt.subplots(figsize=(10,4))
ax.plot(dates, fraction_corr, c='b', label='With mobility')
ax.plot(dates, fraction_corr_nomob, c='black', label = 'Without mobility')
ax.axvline(x = datetime(2020,7,1),c='green')
ax.annotate(xy=(datetime(2020,7,1), setpoint_dates), s="07-01", c='green')
ax.axvline(x = datetime(2020,8,6),c='orange')
ax.annotate(xy=(datetime(2020,8,6), setpoint_dates), s="08-06", c='orange')
ax.axvline(x = datetime(2020,8,18),c='orange')
ax.annotate(xy=(datetime(2020,8,18), setpoint_dates), s="08-18", c='orange')
ax.axvline(x = datetime(2020,9,29),c='orange')
ax.annotate(xy=(datetime(2020,9,29), setpoint_dates), s="09-29", c='orange')
ax.axvline(x = datetime(2020,10,14),c='r')
ax.annotate(xy=(datetime(2020,10,14), setpoint_dates), s="10-14", c='r')
ax.axvline(x = datetime(2020,11,4),c='r')
ax.annotate(xy=(datetime(2020,11,4), setpoint_dates), s="11-04", c='r')
ax.axvline(x = datetime(2020,12,14),c='r')
ax.annotate(xy=(datetime(2020,12,14), setpoint_dates), s="12-14", c='r')
ax.axvline(x = datetime(2020,12,25),c='orange')
ax.annotate(xy=(datetime(2020,12,25), setpoint_dates), s="12-25", c='orange')
ax.xaxis.set_major_locator(months)
ax.set_xlabel('Date')
#ax.set_ylabel('Effective R')
ax.set_title('Spearman correlation between forecast and observed number of reported infections')
ax.legend(loc = 'center left',bbox_to_anchor=(0,0.25))
plt.ylim([ylim_lower,ylim_upper])
plt.show()
fig.savefig('fig_final_frac_corr_'+str(num_days)+'_new.pdf',format="pdf", bbox_inches="tight")
