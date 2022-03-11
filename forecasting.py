import os
import pandas as pd
from rivm_loader import rivm
from constants import CoronaConstants
import datetime as dt
from mobility_seir import *
import copy


'''
Create forecasts without confidence intervals.
Always check for inclusion of mobility or not!!!
'''


fn = os.path.join(os.path.dirname(__file__), 'data/COVID-19_prevalentie.json')
df_besmettelijk = pd.read_json(fn).set_index('Date')

Avg_contacts = CoronaConstants.contacts_average


# State starting date and horizon:
start_date = "01-07-2020"
end_date = "31-12-2020"

date_format = '%d-%m-%Y'
start_date_number = dt.datetime.strptime(start_date,date_format)
end_date_number = dt.datetime.strptime(end_date,date_format)

Look_into_past = 7


Trans_rates = pd.read_csv("results_forecasting/Trans_rates_NB_mob_final_new.csv").set_index('date')
horizon = 14
horizon_2 = 7

Trans_rates['local_avg7'] = Trans_rates['rate_loc'].rolling(7).mean().shift(0)
Trans_rates['mob_avg7'] = Trans_rates['rate_mob'].rolling(7).mean().shift(0)
#Trans_rates['mob_avg7'] = 0
Trans_rates['pos_avg7'] = Trans_rates['frac_pos'].rolling(7).mean().shift(0)
Trans_rates['mobility_avg7'] = Trans_rates['mobility'].rolling(7).mean().shift(0)



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

columns_2 = ['name','date','pred','outcome','diff','rel_diff']
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
    init_df_02 = rivm.SEI2R2_init(mmdd=RIVM_date,  undetected_multiplier=1 / CoronaConstants.fraction_tested)
    
    HELP_df = init_df_02
    HELP_df["date"] = RIVM_date_str  
    HELP_df = HELP_df.reset_index()
    HELP_df = HELP_df.set_index(['index','date']).rename(columns = {'index':'name'})
    dfs_RIVM = dfs_RIVM.append(HELP_df)
     
  

    #Set new constants:
        
        
    CoronaConstants.contacts_average = Avg_contacts
    current_row = Trans_rates.loc[RIVM_date_str]

        

    
 

    rate_loc = current_row['local_avg7']
    rate_mob = current_row['mob_avg7']
    frac_pos = current_row['frac_pos']
    CoronaConstants.average_total_mobility = current_row['mobility_avg7']
    
    #Compute epsilon, fraction p
    Fraction_local_contacts = 1 / (2*CoronaConstants.average_total_mobility / CoronaConstants.population_nl *rate_mob / rate_loc + 1)
    transmission_prob = rate_loc / CoronaConstants.contacts_average / Fraction_local_contacts
    #transmission_prob = rate_loc / Fraction_local_contacts
    
    #CoronaConstants = CoronaConstants(fraction_local_contacts = Fraction_local_contacts,
    #                                  transmission_prob = transmission_prob)
    CoronaConstants.fraction_local_contacts = Fraction_local_contacts
    CoronaConstants.transmission_prob = transmission_prob
    
    #CoronaConstants = CoronaConstants(**{'transmission_prob':transmission_prob})
    
                         
    
    seir_model = MobilitySEIR(init_df_02,horizon = horizon + Look_into_past, start_date = Current_date_number.strftime('%d-%m-%Y'),
                              time_dependency = False,
                              constants = CoronaConstants(transmission_prob = transmission_prob))

    seir_model.simulate_all()
    Current_end_date_number = Current_date_number + dt.timedelta(days = horizon)
    
    Current_end_date = Current_end_date_number.strftime('%d-%m-%Y')
    
    HELP_df = seir_model.state_at_horizon()
    HELP_df = HELP_df.reset_index()
    HELP_df['date'] = Current_end_date
    HELP_df = HELP_df.set_index(['name','date'])
    
    
    #Compute horizon comparison
    Pred_state_begin = seir_model.state_at_time(Look_into_past)
    Pred_state_end = seir_model.state_at_time(horizon + Look_into_past)
    #if r == r_eff_estimate and mob_red == 1-fraction_mobility:
        
    
    mmdd_end = rivm.date2mmdd(Current_date_number + dt.timedelta(days = horizon) )
    mmdd_begin = rivm.date2mmdd(Current_date_number)
    df_pred_It = Pred_state_end['infected_tested'] + Pred_state_end['removed_tested'] - Pred_state_begin['infected_tested'] - Pred_state_begin['removed_tested']
    df_real = rivm.SEI2R2_init(mmdd_end) - rivm.SEI2R2_init(mmdd_begin)
    df_real_It = df_real['infected_tested'] + df_real['removed_tested']
    df_diff = abs(df_pred_It - df_real_It)
    df_rel_diff = abs(df_pred_It - df_real_It)/df_real_It * 100
    data = [
    [
        a,
        Current_date,
        df_pred_It[a],
        df_real_It[a],
        df_diff[a],
        df_rel_diff[a]
    ]
    for a,x in df_pred_It.items()
    ]
    columns_2 = ['name','date','pred','outcome','diff','rel_diff']


    df_new = pd.DataFrame(data=data,columns=columns_2)
    df_return = df_return.append(df_new)
    
    
        #Compute horizon comparison
    Pred_state_begin = seir_model.state_at_time(Look_into_past)
    Pred_state_end = seir_model.state_at_time(horizon_2 + Look_into_past)
 
    
    mmdd_end = rivm.date2mmdd(Current_date_number + dt.timedelta(days = horizon_2) )
    mmdd_begin = rivm.date2mmdd(Current_date_number)
    df_pred_It = Pred_state_end['infected_tested'] + Pred_state_end['removed_tested'] - Pred_state_begin['infected_tested'] - Pred_state_begin['removed_tested']
    df_real = rivm.SEI2R2_init(mmdd_end) - rivm.SEI2R2_init(mmdd_begin)
    df_real_It = df_real['infected_tested'] + df_real['removed_tested']
    df_diff = abs(df_pred_It - df_real_It)
    df_rel_diff = abs(df_pred_It - df_real_It)/df_real_It * 100
    data = [
    [
        a,
        Current_date,
        df_pred_It[a],
        df_real_It[a],
        df_diff[a],
        df_rel_diff[a]
    ]
    for a,x in df_pred_It.items()
    ]
    columns_2 = ['name','date','pred','outcome','diff','rel_diff']

    df_new_2 = pd.DataFrame(data=data,columns=columns_2)
    df_return_2 = df_return_2.append(df_new_2)
    
    
                    #Compute horizon comparison (today)
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
        df_pred_It[a],
        df_real_It[a],
        df_diff[a],
        df_rel_diff[a]
    ]
    for a,x in df_pred_It.items()
    ]
    columns_2 = ['name','date','pred','outcome','diff','rel_diff']

    df_new_today = pd.DataFrame(data=data,columns=columns_2)
    df_return_today = df_return_today.append(df_new_today)

    
    
    dfs_pred = dfs_pred.append(HELP_df)
    dfs_pred_daily[Current_end_date] = seir_model.daily_reported_infections()
