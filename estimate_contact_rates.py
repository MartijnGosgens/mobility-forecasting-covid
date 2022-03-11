import os
import pandas as pd
from rivm_loader import rivm
from constants import CoronaConstants
import datetime as dt
from mobility_seir import *
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import optimize

fn = os.path.join(os.path.dirname(__file__), 'data/COVID-19_prevalentie.json')
df_besmettelijk = pd.read_json(fn).set_index('Date')



def estimate_rates(start_date,end_date):
    
    date_format = '%d-%m-%Y'
    start_date_number = dt.datetime.strptime(start_date,date_format)
    end_date_number = dt.datetime.strptime(end_date,date_format)
    
    contacts_loc = []
    contacts_mob = []
    contacts_loc_lin = []
    contacts_mob_lin = []
    fraction_positive = []
    
    Transmission_rates_all = pd.DataFrame(columns = ['date','rate_loc','rate_mob','frac_pos']).set_index('date')
    
    for t in range(0, (end_date_number - start_date_number).days + 1):
        day_difference = dt.timedelta(days = t)
        Current_date_number = start_date_number + day_difference
        Current_date = Current_date_number.strftime('%d-%m-%Y')
        RIVM_date = rivm.date2mmdd(Current_date_number)
        
        
        #Create initializations of today and tomorrow
        init_df = rivm.SEI2R2_init(RIVM_date)
        Num_tested = sum(init_df["infected_tested"])
        Est_besmettelijk = df_besmettelijk.at[Current_date_number,'prev_avg']
        CoronaConstants.fraction_tested = Num_tested / Est_besmettelijk
        init_df_02 = rivm.SEI2R2_init(mmdd=RIVM_date,  undetected_multiplier=1 / CoronaConstants.fraction_tested)
        
        HELP_frac = CoronaConstants.fraction_tested
        fraction_positive.append(CoronaConstants.fraction_tested)
        
        Next_date_number = Current_date_number + dt.timedelta(1)
        RIVM_date_next = rivm.date2mmdd(Next_date_number)
        init_df_next = rivm.SEI2R2_init(RIVM_date_next)
        Num_tested_next = sum(init_df_next["infected_tested"])
        Est_besmettelijk_next = df_besmettelijk.at[Next_date_number,'prev_avg']    
        init_df_next_02 = rivm.SEI2R2_init(mmdd=RIVM_date_next,  undetected_multiplier=1 / CoronaConstants.fraction_tested)
    

        #Construct regression terms        
        Regression_outcome = init_df_02["susceptible"] - init_df_next_02["susceptible"]
        seir_model = MobilitySEIR(init_df_02,horizon = 1, time_dependency = False, start_date = Current_date_number.strftime('%d-%m-%Y'))
        seir_model.simulate_all_contacts()
        Regression_coefs = seir_model.coeffs_at_horizon()
    
        Regression_coefs["outcome"] = Regression_outcome
        #Remove municipalities that do not have any infections
        Regression_coefs = Regression_coefs.loc[(Regression_coefs["coeff_loc"] > 0) | (Regression_coefs["coeff_mob"] > 0)]
        
        B_loc = Regression_coefs["coeff_loc"]
        B_mob = Regression_coefs["coeff_mob"]
        Delta_S = Regression_coefs["outcome"]
        
        Num_gemeentes = len(Delta_S)
        
        #Define log-MLE function
        def fun_log(x):
            return sum(max(0,Delta_S[i]) * -1 * np.log(x[0]*B_loc[i] + x[1]*B_mob[i]) 
                       +x[0]*B_loc[i] + x[1]*B_mob[i] 
                       for i in range(0,len(Delta_S)))
        
        def fun_lsq(x):
            return sum((max(0,Delta_S[i]) - x[0]*B_loc[i] - x[1]*B_mob[i] )**2
                       for i in range(0,len(Delta_S)))
        
        
        #Minimize the MLE-function
        estimated_rates = optimize.minimize(fun_log, [1,1], method = "SLSQP")        
        new_row = {'date':Current_date,'rate_loc':estimated_rates.x[0],'rate_mob':estimated_rates.x[1],'frac_pos':HELP_frac}
        Transmission_rates_all = Transmission_rates_all.append(new_row, ignore_index = True)
        
    return Transmission_rates_all