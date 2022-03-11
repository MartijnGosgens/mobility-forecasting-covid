import os
import pandas as pd
from rivm_loader import rivm
from constants import CoronaConstants
import datetime as dt
from mobility_seir import *
import numpy as np
from scipy import optimize, special
import matplotlib.pyplot as plt
import math
from mobility import Mobility
from datetime import datetime
import matplotlib.dates as mdates
from mezuro_preprocessing import gemeente2gemeente
import time

np.random.seed(42)

fn = os.path.join(os.path.dirname(__file__), 'data/COVID-19_prevalentie.json')
df_besmettelijk = pd.read_json(fn).set_index('Date')

gemeente_HELP = (gemeente2gemeente.reset_index()).groupby('datum')

#Estimate rates together with bootstrapping samples and outcomes
'''
Estimate transmission rates from start_date up to and including end_date.

confidence: 
    if True, also compute confidence intervals via bootstrapping
conf_level: 
    % of confidence interval
nr_bootstrap_samples:
    If confidence is True: nr. of bootstrap samples used
    
Warning: computing confidence intervals takes much time! (few seconds per sample)
'''


def estimate_rates(start_date,end_date,confidence = False,conf_level = .95, nr_bootstrap_samples = 100):
    date_format = '%d-%m-%Y'
    start_date_number = dt.datetime.strptime(start_date,date_format)
    end_date_number = dt.datetime.strptime(end_date,date_format)
    Transmission_rates_all = pd.DataFrame(columns = ['date','rate_loc','rate_mob','frac_pos','frac_loc','prob_times_contacts','mobility']).set_index('date')
    Transmission_rates_all_no_mob = pd.DataFrame(columns = ['date','rate_loc','frac_pos']).set_index('date')
    Transmission_rates_CI = pd.DataFrame(columns = ['date','lower_loc','upper_loc','lower_mob','upper_mob','lower_fracloc','upper_fracloc','lower_newlyinfected','upper_newlyinfected']).set_index('date')
    Transmission_rates_CI_no_mob = pd.DataFrame(columns = ['date','lower_loc','upper_loc','lower_newlyinfected','upper_newlyinfected']).set_index('date')
    Transmission_rates_NB = pd.DataFrame(columns = ['date','rate_loc','rate_mob','dispersion','frac_pos','frac_loc','prob_times_contacts','mobility']).set_index('date')
    Transmission_rates_CI_NB = pd.DataFrame(columns = ['date','lower_loc','upper_loc','lower_mob','upper_mob','lower_fracloc','upper_fracloc','lower_newlyinfected','upper_newlyinfected','lower_disp','upper_disp']).set_index('date')
    Transmission_rates_NB_no_mob = pd.DataFrame(columns = ['date','rate_loc','dispersion','frac_pos','prob_times_contacts','mobility']).set_index('date')
    Transmission_rates_CI_NB_no_mob = pd.DataFrame(columns = ['date','lower_loc','upper_loc','lower_newlyinfected','upper_newlyinfected','lower_disp','upper_disp']).set_index('date')
 
    CI_frame_NB = pd.DataFrame(columns = ['date','sample','rate_loc','rate_mob','frac_loc','newlyinfected','R']).set_index('sample')

    
 
    CI_intervals = {}
    CI_intervals_NB = {}
    CI_intervals_no_mob = {}
    CI_intervals_NB_no_mob = {}
    
    dates = []
    
    for t in range(0, (end_date_number - start_date_number).days + 1):
        day_difference = dt.timedelta(days = t)
        Current_date_number = start_date_number + day_difference
        dates.append(Current_date_number)
        Current_date = Current_date_number.strftime('%d-%m-%Y')
        RIVM_date = rivm.date2mmdd(Current_date_number)
        
        
        #Compute total mobility
        CoronaConstants.average_total_mobility = sum((gemeente_HELP.get_group(Current_date))['totaal_aantal_bezoekers'])
        
        #Create initializations of today and tomorrow
        init_df = rivm.SEI2R2_init(RIVM_date)
        Num_tested = sum(init_df["infected_tested"])
        if Current_date_number not in df_besmettelijk.index:
            Last_date = df_besmettelijk.index[-1].strftime('%d-%m-%Y')
            Est_besmettelijk = (df_besmettelijk.at[Last_date,'prev_low'] + df_besmettelijk.at[Last_date,'prev_up'])/2
            print(Last_date)
        else:
            Est_besmettelijk = (df_besmettelijk.at[Current_date_number,'prev_low'] + df_besmettelijk.at[Current_date_number,'prev_up'])/2
        CoronaConstants.fraction_tested = Num_tested / Est_besmettelijk

        
        init_df_02 = rivm.SEI2R2_init(RIVM_date,  undetected_multiplier=1 / CoronaConstants.fraction_tested)
        HELP_frac = CoronaConstants.fraction_tested
        
        Next_date_number = Current_date_number + dt.timedelta(1)
        RIVM_date_next = rivm.date2mmdd(Next_date_number)
        init_df_next = rivm.SEI2R2_init(RIVM_date_next)
        init_df_next_02 = rivm.SEI2R2_init(mmdd=RIVM_date_next,  undetected_multiplier=1 / CoronaConstants.fraction_tested)
    

        #Construct regression terms        
        Regression_outcome = init_df_02["susceptible"] - init_df_next_02["susceptible"]
        seir_model = MobilitySEIR(init_df_02,horizon = 1, time_dependency = True, start_date = Current_date_number.strftime('%d-%m-%Y'))
        seir_model.simulate_all_contacts()
        Regression_coefs = seir_model.coeffs_at_horizon()
    
        Regression_coefs["outcome"] = Regression_outcome
        #Remove municipalities that do not have any infections
        
        Regression_coefs = Regression_coefs.loc[(Regression_coefs["coeff_loc"] > 0)]        
        
        B_loc = Regression_coefs["coeff_loc"]
        B_mob = Regression_coefs["coeff_mob"]
        Delta_S = Regression_coefs["outcome"]

        
        #Define log-MLE function
        def fun_log(x):
            return sum(max(0,Delta_S[i]) * -1 * np.log(x[0]*B_loc[i] + x[1]*B_mob[i]) 
                       +x[0]*B_loc[i] + x[1]*B_mob[i] 
                       + special.loggamma(max(0,Delta_S[i]) + 1)
                       for i in range(0,len(Delta_S)))
             
        def fun_log_no_mob(x):
            return sum(max(0,Delta_S[i]) * -1 * np.log(x*B_loc[i]) 
                       +x*B_loc[i] 
                       + special.loggamma(max(0,Delta_S[i]) + 1)
                       for i in range(0,len(Delta_S)))
        
        def fun_NB(x):
            return sum(
                max(0,Delta_S[i]) * -1 *  np.log(x[0]*B_loc[i] + x[1]*B_mob[i])
                - special.loggamma(x[2] + max(0,Delta_S[i]))
                + special.loggamma(x[2])
                + max(0,Delta_S[i]) * np.log(x[2] + x[0]*B_loc[i] + x[1]*B_mob[i])
                + x[2] * np.log(1 + (x[0]*B_loc[i]+x[1]*B_mob[i]) / x[2])
                + special.loggamma(max(0,Delta_S[i]) + 1)
                for i in range(0,len(Delta_S)))
        
        def fun_NB_no_mob(x):
            return sum(
                max(0,Delta_S[i]) * -1 *  np.log(x[0]*B_loc[i])
                - special.loggamma(x[1] + max(0,Delta_S[i]))
                + special.loggamma(x[1])
                + max(0,Delta_S[i]) * np.log(x[1] + x[0]*B_loc[i])
                + x[1] * np.log(1 + (x[0]*B_loc[i]) / x[1])
                + special.loggamma(max(0,Delta_S[i]) + 1)
                for i in range(0,len(Delta_S)))
        
        #Minimize the MLE-functions
        estimated_rates = optimize.minimize(fun_log, [.1,.1], method = 'Nelder-Mead',bounds = ([0,None],[0,None]))  
        estimated_rates_no_mob = optimize.minimize(fun_log_no_mob, 1, method = 'Nelder-Mead')        
        estimated_rates_NB = optimize.minimize(fun_NB,[.1,.1,20], method = 'Nelder-Mead', bounds = ([0,None],[0,None],[0,None]))
        estimated_rates_NB_no_mob = optimize.minimize(fun_NB_no_mob, [.1,.1], method = 'Nelder-Mead', bounds = ([0,None],[0,None]))
        
        Mobility_graph = Mobility.GemeenteMobility(Current_date)
        division_no_restrictions = {
            area: 0
            for area in init_df.index
        }
        CoronaConstants.average_total_mobility = Mobility_graph.intra_region_mobility(division_no_restrictions)
        
        B_loc_init = max(0,estimated_rates.x[0])
        B_mob_init = max(0,estimated_rates.x[1])
        
        B_loc_init_no_mob = max(0,estimated_rates.x[0])
        
        B_loc_init_NB = max(0,estimated_rates_NB.x[0])
        B_mob_init_NB = max(0,estimated_rates_NB.x[1])
        Dispersion_init_NB = max(0,estimated_rates_NB.x[2])
        
        B_loc_init_NB_no_mob = max(0,estimated_rates_NB_no_mob.x[0])
        Dispersion_init_NB_no_mob = max(0,estimated_rates_NB_no_mob.x[1])
        
        Fraction_local_contacts_init = 1 / (2*CoronaConstants.average_total_mobility / CoronaConstants.population_nl *B_mob_init / B_loc_init + 1)
        
        

        #Compute fraction_local        
        if Fraction_local_contacts_init == 0:
            Newlyinfected_init = B_mob_init * 2 * CoronaConstants.average_total_mobility / CoronaConstants.population_nl
        else:
            Newlyinfected_init = B_loc_init / Fraction_local_contacts_init

            
        Fraction_local_contacts_init_NB = 1 / (2*CoronaConstants.average_total_mobility / CoronaConstants.population_nl *B_mob_init_NB / B_loc_init_NB + 1)
        if Fraction_local_contacts_init_NB == 0:
            Newlyinfected_init_NB = B_mob_init_NB * 2 *  CoronaConstants.average_total_mobility / CoronaConstants.population_nl
        else:
            Newlyinfected_init_NB = B_loc_init_NB / Fraction_local_contacts_init_NB
        
        
        #AIC:
        #Compute value needed for GoF values:
        GoF_value_log = 4 + 2 * fun_log([B_loc_init,B_mob_init])
        GoF_value_no_mob = 2 + 2 * fun_log_no_mob(B_loc_init_no_mob)
        GoF_value_NB = 6 + 2 * fun_NB([B_loc_init_NB,B_mob_init_NB,Dispersion_init_NB])
        GoF_value_NB_no_mob = 4 + 2 * fun_NB_no_mob([B_loc_init_NB_no_mob,Dispersion_init_NB_no_mob])

        #Compute effective R for mobility-case
        CoronaConstants.transmission_prob = Newlyinfected_init / CoronaConstants.contacts_average
        CoronaConstants.fraction_local_contacts = Fraction_local_contacts_init
        CoronaConstants.contacts_local = CoronaConstants.fraction_local_contacts*CoronaConstants.contacts_average   # rho*c
        CoronaConstants.contacts_per_visit = (1-CoronaConstants.fraction_local_contacts)*CoronaConstants.contacts_average*CoronaConstants.population_nl/(2*CoronaConstants.average_total_mobility) # c_m : see supplementary material


        seir_for_R = MobilitySEIR(init_df_02,horizon = 1, time_dependency = True, start_date = Current_date_number.strftime('%d-%m-%Y'),
                                  constants = CoronaConstants)
        Current_R = seir_for_R.effective_reproduction_number()
        
        Current_R_no_mob = 0
        
        #Compute effective R for no-mobility-case
        CoronaConstants.transmission_prob = estimated_rates_no_mob.x / CoronaConstants.contacts_average
        CoronaConstants.fraction_local_contacts = 1
        CoronaConstants.contacts_local = CoronaConstants.fraction_local_contacts*CoronaConstants.contacts_average   # rho*c
        CoronaConstants.contacts_per_visit = (1-CoronaConstants.fraction_local_contacts)*CoronaConstants.contacts_average*CoronaConstants.population_nl/(2*CoronaConstants.average_total_mobility) # c_m : see supplementary material

        seir_for_R = MobilitySEIR(init_df_02,horizon = 1, time_dependency = True, start_date = Current_date_number.strftime('%d-%m-%Y'),
                                  constants = CoronaConstants)
        Current_R_no_mob = seir_for_R.effective_reproduction_number()
      
        #Compute effective R for NB-case
        CoronaConstants.transmission_prob = Newlyinfected_init_NB / CoronaConstants.contacts_average
        CoronaConstants.fraction_local_contacts = Fraction_local_contacts_init_NB
        CoronaConstants.contacts_local = CoronaConstants.fraction_local_contacts*CoronaConstants.contacts_average   # rho*c
        CoronaConstants.contacts_per_visit = (1-CoronaConstants.fraction_local_contacts)*CoronaConstants.contacts_average*CoronaConstants.population_nl/(2*CoronaConstants.average_total_mobility) # c_m : see supplementary material

        seir_for_R = MobilitySEIR(init_df_02,horizon = 1, time_dependency = True, start_date = Current_date_number.strftime('%d-%m-%Y'),
                                  constants = CoronaConstants)
        Current_R_NB = seir_for_R.effective_reproduction_number()

        
        #Compute effective R for NB-case with no mobility
        CoronaConstants.transmission_prob = estimated_rates_NB_no_mob.x[0] / CoronaConstants.contacts_average
        CoronaConstants.fraction_local_contacts = 1
        CoronaConstants.contacts_local = CoronaConstants.fraction_local_contacts*CoronaConstants.contacts_average   # rho*c
        CoronaConstants.contacts_per_visit = (1-CoronaConstants.fraction_local_contacts)*CoronaConstants.contacts_average*CoronaConstants.population_nl/(2*CoronaConstants.average_total_mobility) # c_m : see supplementary material

        seir_for_R = MobilitySEIR(init_df_02,horizon = 1, time_dependency = True, start_date = Current_date_number.strftime('%d-%m-%Y'),
                                  constants = CoronaConstants)
        Current_R_NB_no_mob = seir_for_R.effective_reproduction_number()
  
        
        #Save data
        new_row = {'date':Current_date,'rate_loc':B_loc_init,'rate_mob':B_mob_init,'frac_pos':HELP_frac,'frac_loc':Fraction_local_contacts_init,'prob_times_contacts':Newlyinfected_init,'mobility':CoronaConstants.average_total_mobility,'R':Current_R,'GoF':GoF_value_log}
        Transmission_rates_all = Transmission_rates_all.append(new_row, ignore_index = True)
        new_row = {'date':Current_date,'rate_loc':B_loc_init_no_mob,'frac_pos':HELP_frac,'R':Current_R_no_mob,'GoF':GoF_value_no_mob}
        Transmission_rates_all_no_mob = Transmission_rates_all_no_mob.append(new_row, ignore_index = True)
  
        new_row = {'date':Current_date,'rate_loc':B_loc_init_NB,'rate_mob':B_mob_init_NB,'dispersion':Dispersion_init_NB,'frac_pos':HELP_frac,'frac_loc':Fraction_local_contacts_init_NB,'prob_times_contacts':Newlyinfected_init_NB,'mobility':CoronaConstants.average_total_mobility,'R':Current_R_NB,'GoF':GoF_value_NB}
        Transmission_rates_NB = Transmission_rates_NB.append(new_row, ignore_index = True)
 
        new_row = {'date':Current_date,'rate_loc':B_loc_init_NB_no_mob,'dispersion':Dispersion_init_NB_no_mob,'frac_pos':HELP_frac,'R':Current_R_NB_no_mob,'GoF':GoF_value_NB_no_mob}
        Transmission_rates_NB_no_mob = Transmission_rates_NB_no_mob.append(new_row, ignore_index = True)
    
 
        #Create bootstrapping samples for Poisson
        if confidence:
            CI_frame = pd.DataFrame(columns = ['sample','rate_loc','rate_mob','frac_loc','newlyinfected','R']).set_index('sample')
            CI_frame_no_mob = pd.DataFrame(columns = ['sample','rate_loc','newlyinfected','R']).set_index('sample')
            CI_frame_NB_no_mob = pd.DataFrame(columns = ['sample','rate_loc','newlyinfected','R']).set_index('sample')

            bootstrap_samples = []
            
            print('Find Poisson CI')
            #Estimating CI's for Poisson:
            for sample_nr in range(0,nr_bootstrap_samples):
                print('Sample nr:',sample_nr)
                Regression_outcome_new = []
                #Create new outcomes
                for mun in range(0,len(B_loc)):
                    Regression_outcome_new.append(np.random.poisson(B_loc[mun]*B_loc_init + B_mob[mun]*B_mob_init))
                def fun_log(x):
                    return sum(max(0,Regression_outcome_new[i]) * -1 * np.log(x[0]*B_loc[i] + x[1]*B_mob[i]) 
                           +x[0]*B_loc[i] + x[1]*B_mob[i] 
                           for i in range(0,len(Regression_outcome_new)))



                estimated_rates_new = optimize.minimize(fun_log, [.1,.1], method = "Nelder-Mead",bounds = ([0,None],[0,None]))  



                fracloc_new = 1 / (2*CoronaConstants.average_total_mobility / CoronaConstants.population_nl * estimated_rates_new.x[1] / estimated_rates_new.x[0]  + 1)
                if fracloc_new == 0:
                    newlyinfected_new = estimated_rates_new.x[1] * 2*CoronaConstants.average_total_mobility / CoronaConstants.population_nl
                else:
                    newlyinfected_new = estimated_rates_new.x[1] / fracloc_new
                
                
                #Compute effective R for mobility-case
                CoronaConstants.transmission_prob = newlyinfected_new / CoronaConstants.contacts_average
                CoronaConstants.fraction_local_contacts = fracloc_new
                seir_for_R = MobilitySEIR(init_df_02,horizon = 1, time_dependency = True, start_date = Current_date_number.strftime('%d-%m-%Y'),
                                          constants = CoronaConstants)
                Current_R_new = seir_for_R.effective_reproduction_number_alt()
           

                
                bootstrap_samples.append((estimated_rates_new.x).tolist() + [fracloc_new,newlyinfected_new,Current_R_new])

                
                
                row_new = {'sample': sample_nr, 'rate_loc':estimated_rates_new.x[0], 'rate_mob': estimated_rates_new.x[1], 'frac_loc':fracloc_new,'newlyinfected':newlyinfected_new,'R':Current_R_new}
                CI_frame = CI_frame.append(row_new, ignore_index = True)
            
            CI_intervals[Current_date] = CI_frame
            
            bootstrap_sample_loc = [bootstrap_samples[i][0] for i in range(0,nr_bootstrap_samples)]
            bootstrap_sample_mob = [bootstrap_samples[i][1] for i in range(0,nr_bootstrap_samples)]     
            bootstrap_sample_fracloc = [bootstrap_samples[i][2] for i in range(0,nr_bootstrap_samples)]
            bootstrap_sample_newlyinfected = [bootstrap_samples[i][3] for i in range(0,nr_bootstrap_samples)]     
            bootstrap_sample_R = [bootstrap_samples[i][4] for i in range(0,nr_bootstrap_samples)]
            
            #Find appropriate quantiles:
            conf_lower = (1 - conf_level) / 2
            conf_upper = (1 + conf_level) / 2
            quant_loc_lower = np.quantile(bootstrap_sample_loc,conf_lower)
            quant_loc_upper = np.quantile(bootstrap_sample_loc,conf_upper)
            quant_mob_lower = np.quantile(bootstrap_sample_mob,conf_lower)
            quant_mob_upper = np.quantile(bootstrap_sample_mob,conf_upper)
            quant_fracloc_lower = np.quantile(bootstrap_sample_fracloc,conf_lower)
            quant_fracloc_upper = np.quantile(bootstrap_sample_fracloc,conf_upper)
            quant_newlyinfected_lower = np.quantile(bootstrap_sample_newlyinfected,conf_lower)
            quant_newlyinfected_upper = np.quantile(bootstrap_sample_newlyinfected,conf_upper)
            quant_R_lower = np.quantile(bootstrap_sample_R,conf_lower)
            quant_R_upper = np.quantile(bootstrap_sample_R,conf_upper)
      
            
            new_row = {'date':Current_date,'lower_loc':quant_loc_lower,'upper_loc':quant_loc_upper,'lower_mob':quant_mob_lower,'upper_mob':quant_mob_upper,
                       'lower_fracloc':quant_fracloc_lower,'upper_fracloc':quant_fracloc_upper,
                       'lower_newlyinfected':quant_newlyinfected_lower,'upper_newlyinfected':quant_newlyinfected_upper,
                       'lower_R':quant_R_lower,'upper_R':quant_R_upper}
            Transmission_rates_CI = Transmission_rates_CI.append(new_row,ignore_index = True)



            print('Find Poisson CI without mobility')
            #Estimating CI's for Poisson:
            for sample_nr in range(0,nr_bootstrap_samples):
                print('Sample nr:',sample_nr)
                Regression_outcome_new = []
                #Create new outcomes
                for mun in range(0,len(B_loc)):
                    Regression_outcome_new.append(np.random.poisson(B_loc[mun]*B_loc_init_no_mob))
                def fun_log_no_mob(x):
                    return sum(max(0,Regression_outcome_new[i]) * -1 * np.log(x[0]*B_loc[i] ) 
                           +x[0]*B_loc[i]
                           for i in range(0,len(Regression_outcome_new)))
                HELP_time_1 = time.time()
                estimated_rates_no_mob_new = optimize.minimize(fun_log_no_mob, 1, method = "Nelder-Mead"  )
                HELP_time_2 = time.time()
                print('Solve time is',HELP_time_2 - HELP_time_1)
                newlyinfected_new = estimated_rates_no_mob_new.x[0]
                
                CoronaConstants.transmission_prob = estimated_rates_no_mob_new.x / CoronaConstants.contacts_average
                CoronaConstants.fraction_local_contacts = 1
                seir_for_R = MobilitySEIR(init_df_02,horizon = 1, time_dependency = True, start_date = Current_date_number.strftime('%d-%m-%Y'),
                                  constants = CoronaConstants)
                Current_R_no_mob_new = seir_for_R.effective_reproduction_number_alt()
                
                bootstrap_samples.append((estimated_rates_no_mob_new.x).tolist() + [newlyinfected_new, Current_R_no_mob_new])
 
                
                row_new = {'sample': sample_nr, 'rate_loc':estimated_rates_no_mob_new.x[0],'newlyinfected':newlyinfected_new,'R': Current_R_no_mob_new}
                CI_frame_no_mob = CI_frame_no_mob.append(row_new, ignore_index = True)
            
            CI_intervals_no_mob[Current_date] = CI_frame
            
            bootstrap_sample_loc = [bootstrap_samples[i][0] for i in range(0,nr_bootstrap_samples)]
            bootstrap_sample_newlyinfected = [bootstrap_samples[i][1] for i in range(0,nr_bootstrap_samples)]
            bootstrap_sample_R = [bootstrap_samples[i][2] for i in range(0,nr_bootstrap_samples)]
            
            #Find appropriate quantiles:
            conf_lower = (1 - conf_level) / 2
            conf_upper = (1 + conf_level) / 2
            quant_loc_lower = np.quantile(bootstrap_sample_loc,conf_lower)
            quant_loc_upper = np.quantile(bootstrap_sample_loc,conf_upper)
            quant_newlyinfected_lower = np.quantile(bootstrap_sample_newlyinfected,conf_lower)
            quant_newlyinfected_upper = np.quantile(bootstrap_sample_newlyinfected,conf_upper)
            quant_R_lower = np.quantile(bootstrap_sample_R,conf_lower)
            quant_R_upper = np.quantile(bootstrap_sample_R,conf_upper)
            
            
            new_row = {'date':Current_date,'lower_loc':quant_loc_lower,'upper_loc':quant_loc_upper,
                       'lower_newlyinfected':quant_newlyinfected_lower,'upper_newlyinfected':quant_newlyinfected_upper,
                        'lower_R':quant_R_lower,'upper_R':quant_R_upper}
            Transmission_rates_CI_no_mob = Transmission_rates_CI_no_mob.append(new_row,ignore_index = True)
 
            
        
            print('Find NB CI')
            HELP_time_1 = time.time()    
            #Estimating CI's for NB:
            bootstrap_samples = []
            for sample_nr in range(0,nr_bootstrap_samples):
                print('Sample nr:',sample_nr)
                Regression_outcome_new = []
                #Create new outcomes
                for mun in range(0,len(B_loc)):
                    p = (B_loc[mun]*B_loc_init_NB + B_mob[mun]*B_mob_init_NB) / (Dispersion_init_NB + B_loc[mun]*B_loc_init_NB + B_mob[mun]*B_mob_init_NB)
                    Regression_outcome_new.append(np.random.negative_binomial(Dispersion_init_NB,1-p))
                def fun_NB(x):
                    return sum(
                        max(0,Regression_outcome_new[i]) * -1 *  np.log(x[0]*B_loc[i] + x[1]*B_mob[i])
                        - special.loggamma(x[2] + max(0,Regression_outcome_new[i]))
                        + special.loggamma(x[2])
                        + max(0,Regression_outcome_new[i]) * np.log(x[2] + x[0]*B_loc[i] + x[1]*B_mob[i])
                        + x[2] * np.log(1 + (x[0]*B_loc[i]+x[1]*B_mob[i]) / x[2])
                        for i in range(0,len(Regression_outcome_new)))
                
                
            
                estimated_rates_new = optimize.minimize(fun_NB, [.1,.1,20], method = "Nelder-Mead",bounds = ([0,None],[0,None],[0,None]))  

                
                
                #bootstrap_samples.append(estimated_rates_new.x)
                fracloc_new = 1 / (2*CoronaConstants.average_total_mobility / CoronaConstants.population_nl * estimated_rates_new.x[1] / estimated_rates_new.x[0]  + 1)
                if fracloc_new == 0:
                    newlyinfected_new = 2*CoronaConstants.average_total_mobility / CoronaConstants.population_nl * estimated_rates_new.x[1] 
                else:
                    newlyinfected_new = estimated_rates_new.x[0] / fracloc_new
                
                CoronaConstants.transmission_prob = newlyinfected_new / CoronaConstants.contacts_average
                CoronaConstants.fraction_local_contacts = fracloc_new
                CoronaConstants.contacts_local = CoronaConstants.fraction_local_contacts*CoronaConstants.contacts_average   # rho*c
                CoronaConstants.contacts_per_visit = (1-CoronaConstants.fraction_local_contacts)*CoronaConstants.contacts_average*CoronaConstants.population_nl/(2*CoronaConstants.average_total_mobility) # c_m : see supplementary material

                seir_for_R = MobilitySEIR(init_df_02,horizon = 1, time_dependency = True, start_date = Current_date_number.strftime('%d-%m-%Y'),
                                  constants = CoronaConstants)
                Current_R_NB_new = seir_for_R.effective_reproduction_number()

                bootstrap_samples.append((estimated_rates_new.x).tolist() + [fracloc_new, newlyinfected_new, Current_R_NB_new])
                
                row_new = {'date':Current_date,'sample': sample_nr, 'rate_loc':estimated_rates_new.x[0], 'rate_mob': estimated_rates_new.x[1], 'frac_loc':fracloc_new,'newlyinfected':newlyinfected_new,'disp':estimated_rates_new.x[2],'R':Current_R_NB_new}
                CI_frame_NB = CI_frame_NB.append(row_new, ignore_index = True)

                HELP_time_2 = time.time()
                print('Solve time is',HELP_time_2 - HELP_time_1)
            
            CI_intervals_NB[Current_date] = CI_frame_NB
            
            bootstrap_sample_loc = [bootstrap_samples[i][0] for i in range(0,nr_bootstrap_samples)]
            bootstrap_sample_mob = [bootstrap_samples[i][1] for i in range(0,nr_bootstrap_samples)]        
            bootstrap_sample_disp = [bootstrap_samples[i][2] for i in range(0,nr_bootstrap_samples)]
            bootstrap_sample_fracloc = [bootstrap_samples[i][3] for i in range(0,nr_bootstrap_samples)]
            bootstrap_sample_newlyinfected = [bootstrap_samples[i][4] for i in range(0,nr_bootstrap_samples)]
            bootstrap_sample_R = [bootstrap_samples[i][5] for i in range(0,nr_bootstrap_samples)]
            
            
            #Find appropriate quantiles:
            conf_lower = (1 - conf_level) / 2
            conf_upper = (1 + conf_level) / 2
            quant_loc_lower = np.quantile(bootstrap_sample_loc,conf_lower)
            quant_loc_upper = np.quantile(bootstrap_sample_loc,conf_upper)
            quant_mob_lower = np.quantile(bootstrap_sample_mob,conf_lower)
            quant_mob_upper = np.quantile(bootstrap_sample_mob,conf_upper)
            quant_disp_lower = np.quantile(bootstrap_sample_disp,conf_lower)
            quant_disp_upper = np.quantile(bootstrap_sample_disp,conf_upper)
            quant_fracloc_lower = np.quantile(bootstrap_sample_fracloc,conf_lower)
            quant_fracloc_upper = np.quantile(bootstrap_sample_fracloc,conf_upper)
            quant_newlyinfected_lower = np.quantile(bootstrap_sample_newlyinfected,conf_lower)
            quant_newlyinfected_upper = np.quantile(bootstrap_sample_newlyinfected,conf_upper)
            quant_R_lower = np.quantile(bootstrap_sample_R,conf_lower)
            quant_R_upper = np.quantile(bootstrap_sample_R,conf_upper)
 
            new_row = {'date':Current_date,'lower_loc':quant_loc_lower,'upper_loc':quant_loc_upper,'lower_mob':quant_mob_lower,'upper_mob':quant_mob_upper,
                       'lower_fracloc':quant_fracloc_lower,'upper_fracloc':quant_fracloc_upper,
                       'lower_newlyinfected':quant_newlyinfected_lower,'upper_newlyinfected':quant_newlyinfected_upper,
                       'lower_disp':quant_disp_lower,'upper_disp':quant_disp_upper,
                       'lower_R':quant_R_lower,'upper_R':quant_R_upper}
            Transmission_rates_CI_NB = Transmission_rates_CI_NB.append(new_row,ignore_index = True)


            
            print('Find NB CI with no mobility')
            #Estimating CI's for NB:
            bootstrap_samples = []
            for sample_nr in range(0,nr_bootstrap_samples):
                print('Sample nr:',sample_nr)
                Regression_outcome_new = []
                #Create new outcomes
                for mun in range(0,len(B_loc)):
                    p = (B_loc[mun]*B_loc_init_NB_no_mob) / (Dispersion_init_NB_no_mob + B_loc[mun]*B_loc_init_NB_no_mob)
                    Regression_outcome_new.append(np.random.negative_binomial(Dispersion_init_NB_no_mob,1-p))
                def fun_NB(x):
                    return sum(
                        max(0,Regression_outcome_new[i]) * -1 *  np.log(x[0]*B_loc[i] )
                        - special.loggamma(x[1] + max(0,Regression_outcome_new[i]))
                        + special.loggamma(x[1])
                        + max(0,Regression_outcome_new[i]) * np.log(x[1] + x[0]*B_loc[i] )
                        + x[1] * np.log(1 + (x[0]*B_loc[i]) / x[1])
                        for i in range(0,len(Regression_outcome_new)))
                estimated_rates_new = optimize.minimize(fun_NB, [.1,.1], method = "Nelder-Mead",bounds = ([0,None],[0,None]))  

                newlyinfected_new = estimated_rates_new.x[0] / 1
                
                CoronaConstants.transmission_prob = newlyinfected_new / CoronaConstants.contacts_average
                CoronaConstants.fraction_local_contacts = 1
                seir_for_R = MobilitySEIR(init_df_02,horizon = 1, time_dependency = True, start_date = Current_date_number.strftime('%d-%m-%Y'),
                                  constants = CoronaConstants)
                Current_R_NB_no_mob_new = seir_for_R.effective_reproduction_number_alt()
 
                bootstrap_samples.append((estimated_rates_new.x).tolist() + [newlyinfected_new, Current_R_NB_no_mob_new])
                
                row_new = {'sample': sample_nr, 'rate_loc':estimated_rates_new.x[0], 'newlyinfected':newlyinfected_new,'disp':estimated_rates_new.x[1],'R':Current_R_NB_new}
                CI_frame_NB_no_mob = CI_frame_NB_no_mob.append(row_new, ignore_index = True)
            
            CI_intervals_NB_no_mob[Current_date] = CI_frame_NB
            
            bootstrap_sample_loc = [bootstrap_samples[i][0] for i in range(0,nr_bootstrap_samples)]
            bootstrap_sample_disp = [bootstrap_samples[i][1] for i in range(0,nr_bootstrap_samples)]
            bootstrap_sample_newlyinfected = [bootstrap_samples[i][2] for i in range(0,nr_bootstrap_samples)]
            bootstrap_sample_R = [bootstrap_samples[i][3] for i in range(0,nr_bootstrap_samples)]
        
            
            #Find appropriate quantiles:
            conf_lower = (1 - conf_level) / 2
            conf_upper = (1 + conf_level) / 2
            quant_loc_lower = np.quantile(bootstrap_sample_loc,conf_lower)
            quant_loc_upper = np.quantile(bootstrap_sample_loc,conf_upper)
            quant_disp_lower = np.quantile(bootstrap_sample_disp,conf_lower)
            quant_disp_upper = np.quantile(bootstrap_sample_disp,conf_upper)
            quant_newlyinfected_lower = np.quantile(bootstrap_sample_newlyinfected,conf_lower)
            quant_newlyinfected_upper = np.quantile(bootstrap_sample_newlyinfected,conf_upper)
            quant_R_lower = np.quantile(bootstrap_sample_R,conf_lower)
            quant_R_upper = np.quantile(bootstrap_sample_R,conf_upper)
 
            new_row = {'date':Current_date,'lower_loc':quant_loc_lower,'upper_loc':quant_loc_upper,
                       'lower_newlyinfected':quant_newlyinfected_lower,'upper_newlyinfected':quant_newlyinfected_upper,
                       'lower_disp':quant_disp_lower,'upper_disp':quant_disp_upper,
                       'lower_R':quant_R_lower,'upper_R':quant_R_upper}
            Transmission_rates_CI_NB_no_mob = Transmission_rates_CI_NB_no_mob.append(new_row,ignore_index = True)
            
            
            
    return Transmission_rates_all, Transmission_rates_all_no_mob, Transmission_rates_NB, Transmission_rates_NB_no_mob, CI_intervals, CI_intervals_no_mob, CI_frame_NB, CI_intervals_NB_no_mob, Transmission_rates_CI, Transmission_rates_CI_no_mob, Transmission_rates_CI_NB, Transmission_rates_CI_NB_no_mob