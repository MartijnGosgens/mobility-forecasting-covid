'''
Compute forecasts based on estimated transmission rates and bootstrap samples
'''

import os
import pandas as pd
from rivm_loader import rivm
from constants import CoronaConstants
import datetime as dt
from mobility_seir import *
import copy


fn = os.path.join(os.path.dirname(__file__), 'data/COVID-19_prevalentie.json')
df_besmettelijk = pd.read_json(fn).set_index('Date')

Avg_contacts = CoronaConstants.contacts_average


# State starting date and horizon:
start_date = "01-10-2020"
end_date = "31-12-2020"

date_format = '%d-%m-%Y'
start_date_number = dt.datetime.strptime(start_date,date_format)
end_date_number = dt.datetime.strptime(end_date,date_format)

#How many days in the past are taken as input for the current simulation?
Look_into_past = 7

#7- and 14-day ahead forecasts
horizon_2 = 7
horizon = 14

#Nr of bootstrap samples
nr_samples = 100

Trans_rates = pd.read_csv("results_forecasting/Trans_rates_NB_mob_samples_final_new.csv")
Trans_rates_basic = pd.read_csv("results_forecasting/Trans_rates_NB_mob_final_new.csv").set_index(['date'])


Trans_rates_new = pd.DataFrame(columns = [*Trans_rates.columns, 'local_avg7','mob_avg7','pos_avg7','mobility_avg7'])


for sample in range(0,nr_samples):
    Trans_rates_help = Trans_rates.groupby('sample').get_group(sample).set_index('date')
    Trans_rates_help['local_avg7'] = Trans_rates_help['rate_loc'].rolling(7).mean().shift(0)
    Trans_rates_help['mob_avg7'] = Trans_rates_help['rate_mob'].rolling(7).mean().shift(0)
    Trans_rates_help['pos_avg7'] = Trans_rates_basic['frac_pos'].rolling(7).mean().shift(0)
    Trans_rates_help['mobility_avg7'] = Trans_rates_basic['mobility'].rolling(7).mean().shift(0)
    Trans_rates_new = Trans_rates_new.append(Trans_rates_help.reset_index())
print('Done!')
Trans_rates = Trans_rates_new.set_index(['date','sample'])

# Create dataframe dictionaries
dfs_RIVM = pd.DataFrame(columns = ['name','date',
                                   'susceptible','exposed','infected_tested',
                                   'infected_nottested',
                                   'removed_tested','removed_nottested','inhabitant']).set_index(['name','date'])
dfs_pred = pd.DataFrame(columns = ['name','date',
                                   'susceptible','exposed','infected_tested',
                                   'infected_nottested',
                                   'removed_tested','removed_nottested']).set_index(['name','date'])
dfs_pred_daily = {}

columns_2 = ['name','date','sample','pred','outcome','diff','rel_diff']
df_return = pd.DataFrame(columns = columns_2)
df_return_2 = pd.DataFrame(columns = columns_2)
df_return_today = pd.DataFrame(columns = columns_2)

for t in range(0, (end_date_number - start_date_number).days + 1):
    day_difference = dt.timedelta(days = t)
    Current_date_number = start_date_number + day_difference
    Current_date = Current_date_number.strftime('%d-%m-%Y')
    print('Date:',Current_date)
    #Compute right starting date for RIVM simulations
    RIVM_date_number = Current_date_number - dt.timedelta(days = Look_into_past)
    RIVM_date_str = RIVM_date_number.strftime('%d-%m-%Y')
    RIVM_date = rivm.date2mmdd(RIVM_date_number)

    
    init_df = rivm.SEI2R2_init(RIVM_date)
    Num_tested = sum(init_df["infected_tested"])
    Est_besmettelijk = df_besmettelijk.at[Current_date_number,'prev_avg']
    CoronaConstants.fraction_tested = Num_tested / Est_besmettelijk
    #print(CoronaConstants.fraction_tested)
    init_df_02 = rivm.SEI2R2_init(mmdd=RIVM_date,  undetected_multiplier=1 / CoronaConstants.fraction_tested)
    
    HELP_df = init_df_02
    HELP_df["date"] = RIVM_date_str  
    HELP_df = HELP_df.reset_index()
    HELP_df = HELP_df.set_index(['index','date']).rename(columns = {'index':'name'})
    dfs_RIVM = dfs_RIVM.append(HELP_df)
     
  

    #Set new constants:
        
    for sample in range(0,nr_samples):
        print(Current_date, sample)
        CoronaConstants.contacts_average = Avg_contacts
        current_row = Trans_rates.loc[RIVM_date_str,sample]
        print(RIVM_date_str)
            
 
        rate_loc = current_row['local_avg7']
        rate_mob = current_row['mob_avg7']
        frac_pos = current_row['pos_avg7']
        CoronaConstants.average_total_mobility = current_row['mobility_avg7']
        
        #Compute epsilon, fraction p
        Fraction_local_contacts = 1 / (2*CoronaConstants.average_total_mobility / CoronaConstants.population_nl *rate_mob / rate_loc + 1)

        transmission_prob = rate_loc / CoronaConstants.contacts_average / Fraction_local_contacts
        #transmission_prob = rate_loc / Fraction_local_contacts
        
        CoronaConstants.fraction_local_contacts = Fraction_local_contacts
        CoronaConstants.transmission_prob = transmission_prob
        
     
        seir_model = MobilitySEIR(init_df_02,horizon = horizon + Look_into_past, start_date = Current_date_number.strftime('%d-%m-%Y'),
                                  time_dependency = False,
                                  constants = CoronaConstants(transmission_prob = transmission_prob))
        seir_model.simulate_all()
        Current_end_date_number = Current_date_number + dt.timedelta(days = horizon)
        
        Current_end_date = Current_end_date_number.strftime('%d-%m-%Y')
        
        HELP_df = seir_model.state_at_horizon()
        HELP_df = HELP_df.reset_index()
        HELP_df['date'] = Current_end_date
        HELP_df['sample'] = sample
        HELP_df = HELP_df.set_index(['name','date','sample'])
        
        
        #Compute horizon comparison
        Pred_state_begin = seir_model.state_at_time(Look_into_past)
        Pred_state_end = seir_model.state_at_time(horizon + Look_into_past)
         
        
        mmdd_end = rivm.date2mmdd(Current_date_number + dt.timedelta(days = horizon) )
        mmdd_begin = rivm.date2mmdd(Current_date_number)
        df_pred_It = Pred_state_end['infected_tested'] + Pred_state_end['removed_tested'] - Pred_state_begin['infected_tested'] - Pred_state_begin['removed_tested']
       df_real_It = rivm.rivm_corona(mmdd_end) - rivm.rivm_corona(mmdd_begin)
        df_diff = abs(df_pred_It - df_real_It)
        df_rel_diff = abs(df_pred_It - df_real_It)/df_real_It * 100
        data = [
        [
            a,
            Current_date,
            sample,
            df_pred_It[a],
            df_real_It[a],
            df_diff[a],
            df_rel_diff[a]
        ]
        for a,x in df_pred_It.items()
        ]
        columns_2 = ['name','date','sample','pred','outcome','diff','rel_diff']
    
        df_new = pd.DataFrame(data=data,columns=columns_2)
        df_return = df_return.append(df_new)
        
        
            #Compute horizon comparison
        Pred_state_begin = seir_model.state_at_time(Look_into_past)
        Pred_state_end = seir_model.state_at_time(horizon_2 + Look_into_past)
           
        
        mmdd_end = rivm.date2mmdd(Current_date_number + dt.timedelta(days = horizon_2) )
        mmdd_begin = rivm.date2mmdd(Current_date_number)
        df_pred_It = Pred_state_end['infected_tested'] + Pred_state_end['removed_tested'] - Pred_state_begin['infected_tested'] - Pred_state_begin['removed_tested']
        df_real_It = rivm.rivm_corona(mmdd_end) - rivm.rivm_corona(mmdd_begin)
        df_diff = abs(df_pred_It - df_real_It)
        df_rel_diff = abs(df_pred_It - df_real_It)/df_real_It * 100
        data = [
        [
            a,
            Current_date,
            sample,
            df_pred_It[a],
            df_real_It[a],
            df_diff[a],
            df_rel_diff[a]
        ]
        for a,x in df_pred_It.items()
        ]
        columns_2 = ['name','date','sample','pred','outcome','diff','rel_diff']
    
        df_new_2 = pd.DataFrame(data=data,columns=columns_2)
        df_return_2 = df_return_2.append(df_new_2)
        
        
        
        
        
                    #Compute horizon comparison
        Pred_state_begin = seir_model.state_at_time(0)
        Pred_state_end = seir_model.state_at_time(Look_into_past)
     
        mmdd_end = rivm.date2mmdd(Current_date_number)
        mmdd_begin = rivm.date2mmdd(Current_date_number - dt.timedelta(days = Look_into_past))
        df_pred_It = Pred_state_end['infected_tested'] + Pred_state_end['removed_tested'] - Pred_state_begin['infected_tested'] - Pred_state_begin['removed_tested']
        df_real_It = rivm.rivm_corona(mmdd_end) - rivm.rivm_corona(mmdd_begin)
        df_diff = abs(df_pred_It - df_real_It)
        df_rel_diff = abs(df_pred_It - df_real_It)/df_real_It * 100
        data = [
        [
            a,
            Current_date,
            sample,
            df_pred_It[a],
            df_real_It[a],
            df_diff[a],
            df_rel_diff[a]
        ]
        for a,x in df_pred_It.items()
        ]
        columns_2 = ['name','date','sample','pred','outcome','diff','rel_diff']
    
        df_new_today = pd.DataFrame(data=data,columns=columns_2)
        df_return_today = df_return_today.append(df_new_today)
        
        
        
        
        
        
        dfs_pred = dfs_pred.append(HELP_df)
        dfs_pred_daily[Current_end_date] = seir_model.daily_reported_infections()
