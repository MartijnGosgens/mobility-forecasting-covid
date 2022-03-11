import pandas as pd
import copy
from constants import CoronaConstants
from datetime import datetime, timedelta
from mezuro_preprocessing import koppeltabel, gemeente_shapes_old, data_path

# The gemeente naming that is used in the RIVM file
rivm_file_year_gemeenten = '2021'
# The gemeente naming that must be used in the outputed file
year_gemeenten_used = '2020'
# The gemeente naming that is used in the mezuro shape-file:
year_meuro = '2020'

class RivmLoader:
    def __init__(self):
        fn = data_path('COVID-19_aantallen_gemeente_per_dag.csv')
        rivm = pd.read_csv(fn, delimiter=';')
        mmdd2area2reported = {}
        for _, (date, area, reported) in rivm[~rivm.Municipality_name.isnull()][
            ['Date_of_publication', 'Municipality_name', 'Total_reported']].iterrows():
            mmdd = ''.join(date.split('-')[1:]) if date[:4] == '2020' else ''.join(date.split('-'))
            if not mmdd in mmdd2area2reported:
                mmdd2area2reported[mmdd] = {}
            mmdd2area2reported[mmdd][area] = reported
        self.mmdd2area2reported = {
            mmdd: pd.Series(area2reported)
            for mmdd, area2reported in mmdd2area2reported.items()
        }
        cumulative = 0
        mmdd2cumulatives = {}
        for mmdd, infections in self.mmdd2area2reported.items():
            cumulative = infections + cumulative
            mmdd2cumulatives[mmdd] = cumulative
        self.mmdd2cumulatives = mmdd2cumulatives
        
        #For initializing compartments
        self.max_lookback = 14
        self.max_lookforward = 7
        
        #Getting the right gemeente_shape file:
        z = year_gemeenten_used
        inhabitants_new = pd.Series({
            i : gemeente_shapes_old.loc[i,'inhabitant']
            for i in gemeente_shapes_old.index
        })
        
        self.gemeente_inhabitants = pd.Series({
            i : sum(inhabitants_new[j] for j in gemeente_shapes_old.index if koppeltabel.at[j,'rivm'+z] == i)
            for i in list(set(koppeltabel['rivm'+z]))
        })
        

    def rivm_corona(self, mmdd):
        if not mmdd in self.mmdd2cumulatives:
            print(mmdd, 'not in rivm data')
            return 0
        data = self.mmdd2cumulatives[mmdd]
        y = rivm_file_year_gemeenten
        z = year_gemeenten_used



        df_2019 = pd.Series({
        i : data[koppeltabel.loc[i, 'rivm'+y]] * koppeltabel.loc[i, 'inhabitant_frac_'+y]
        for i in gemeente_shapes_old.index
        })  
        return pd.Series({
            i : sum(df_2019[j] for j in gemeente_shapes_old.index if koppeltabel.at[j,'rivm'+z] == i)
            for i in list(set(koppeltabel['rivm'+z]))
            })

    
    def rivm_corona_daily(self, mmdd):
        if not mmdd in self.mmdd2area2reported:
            print(mmdd, 'not in rivm data')
            return 0
        data = self.mmdd2area2reported[mmdd]
        
        y=rivm_file_year_gemeenten
        z = year_gemeenten_used
        
        

    
        
        df_2019 = pd.Series({
        i : data[koppeltabel.loc[i, 'rivm'+y]] * koppeltabel.loc[i, 'inhabitant_frac_'+y]
        for i in gemeente_shapes_old.index
        })  
        return pd.Series({
            i : sum(df_2019[j] for j in gemeente_shapes_old.index if koppeltabel.at[j,'rivm'+z] == i)
            for i in list(set(koppeltabel['rivm'+z]))
            })
        

    def susceptible(self,
                    mmdd,
                    #totals = gemeente_shapes['inhabitant'],
                    undetected_multiplier=1 / CoronaConstants.fraction_tested,
                    latent_period=CoronaConstants.latent_period):
        date = RivmLoader.mmdd2date(mmdd)
        

        
        mmdd_start = RivmLoader.date2mmdd(date + timedelta(1))
 
        df = self.rivm_corona_daily(mmdd_start)
        fraction_still_exposed = 1
        for step in range(2,self.max_lookforward):
            fraction_still_exposed *= (1-1 / latent_period)
            mmdd_current = RivmLoader.date2mmdd(date + timedelta(step))
            df += fraction_still_exposed * self.rivm_corona_daily(mmdd_current)
        
        fraction_remainder = latent_period - (1-pow(1 - 1 / latent_period,self.max_lookback-1)) * latent_period
        mmdd_end = RivmLoader.date2mmdd(date + timedelta(self.max_lookforward))
        df += fraction_remainder * self.rivm_corona_daily(mmdd_end)       
        
        return self.gemeente_inhabitants - undetected_multiplier * self.rivm_corona(mmdd) - undetected_multiplier * df
        
    
     
    def exposed(self,
                mmdd,
                undetected_multiplier=1 / CoronaConstants.fraction_tested,
                latent_period=CoronaConstants.latent_period):
        date = RivmLoader.mmdd2date(mmdd)

        
    
        
        
        
        mmdd_start = RivmLoader.date2mmdd(date + timedelta(1))
 
        df = self.rivm_corona_daily(mmdd_start)
        fraction_still_exposed = 1
        for step in range(2,self.max_lookforward):
            fraction_still_exposed *= (1-1 / latent_period)
            mmdd_current = RivmLoader.date2mmdd(date + timedelta(step))
            df += fraction_still_exposed * self.rivm_corona_daily(mmdd_current)
        
        fraction_remainder = latent_period - (1-pow(1 - 1 / latent_period,self.max_lookback-1)) * latent_period
        mmdd_end = RivmLoader.date2mmdd(date + timedelta(self.max_lookforward))
        df += fraction_remainder * self.rivm_corona_daily(mmdd_end)     
        return undetected_multiplier * df
        

    def infected_tested(self,
                        mmdd,
                        infectious_period=CoronaConstants.infectious_period):
        date = RivmLoader.mmdd2date(mmdd)
        

        df = self.rivm_corona_daily(mmdd)
        fraction_still_infected = 1
        for step in range(1,self.max_lookback):
            fraction_still_infected *= (1-1 / infectious_period)
            mmdd_current = RivmLoader.date2mmdd(date - timedelta(step))
            df += fraction_still_infected * self.rivm_corona_daily(mmdd_current)
        
        fraction_remainder = infectious_period - (1-pow(1 - 1 / infectious_period,self.max_lookback)) * infectious_period
        mmdd_end = RivmLoader.date2mmdd(date - timedelta(self.max_lookback))
        df += fraction_remainder * self.rivm_corona_daily(mmdd_end)     
        return df
        
        
    def infected_nottested(self,
                           mmdd,
                           infectious_period=CoronaConstants.infectious_period,
                           undetected_multiplier=1 / CoronaConstants.fraction_tested):
        date = RivmLoader.mmdd2date(mmdd)
            

        df = self.rivm_corona_daily(mmdd)
        fraction_still_infected = 1
        for step in range(1,self.max_lookback):
            fraction_still_infected *= (1-1 / infectious_period)
            mmdd_current = RivmLoader.date2mmdd(date - timedelta(step))
            df += fraction_still_infected * self.rivm_corona_daily(mmdd_current)
        
        fraction_remainder = infectious_period - (1-pow(1 - 1 / infectious_period,self.max_lookback)) * infectious_period
        mmdd_end = RivmLoader.date2mmdd(date - timedelta(self.max_lookback))
        df += fraction_remainder * self.rivm_corona_daily(mmdd_end)      
        return (undetected_multiplier - 1) * df
        
        
    def removed_tested(self,
                       mmdd,
                       infectious_period=CoronaConstants.infectious_period):
        date = RivmLoader.mmdd2date(mmdd)
        

        df = self.rivm_corona_daily(mmdd)
        fraction_still_infected = 1
        for step in range(1,self.max_lookback):
            fraction_still_infected *= (1-1 / infectious_period)
            mmdd_current = RivmLoader.date2mmdd(date - timedelta(step))
            df += fraction_still_infected * self.rivm_corona_daily(mmdd_current)
        
        fraction_remainder = infectious_period - (1-pow(1 - 1 / infectious_period,self.max_lookback)) * infectious_period
        mmdd_end = RivmLoader.date2mmdd(date - timedelta(self.max_lookback))
        df += fraction_remainder * self.rivm_corona_daily(mmdd_end)      
        return self.rivm_corona(mmdd) - df
        

    def removed_nottested(self,
                          mmdd,
                          infectious_period=CoronaConstants.infectious_period,
                          undetected_multiplier=1 / CoronaConstants.fraction_tested):
        date = RivmLoader.mmdd2date(mmdd)
        

        df = self.rivm_corona_daily(mmdd)
        fraction_still_infected = 1
        for step in range(1,self.max_lookback):
            fraction_still_infected *= (1-1 / infectious_period)
            mmdd_current = RivmLoader.date2mmdd(date - timedelta(step))
            df += fraction_still_infected * self.rivm_corona_daily(mmdd_current)
        
        fraction_remainder = infectious_period - (1-pow(1 - 1 / infectious_period,self.max_lookback)) * infectious_period
        mmdd_end = RivmLoader.date2mmdd(date - timedelta(self.max_lookback))
        df += fraction_remainder * self.rivm_corona_daily(mmdd_end)      
        return (undetected_multiplier - 1) * (self.rivm_corona(mmdd) - df)
        
        
    def SEI2R2_init(self,
                    mmdd='0421',
                    infectious_period=CoronaConstants.infectious_period,
                    undetected_multiplier=1 / CoronaConstants.fraction_tested,
                    latent_period=CoronaConstants.latent_period,
                    return_integer=False
                    ):
        df = pd.DataFrame()
        df["susceptible"] = self.susceptible(mmdd=mmdd, latent_period=latent_period,
                                             undetected_multiplier=undetected_multiplier)
        df["exposed"] = self.exposed(mmdd,undetected_multiplier = undetected_multiplier,latent_period=latent_period)
        df["infected_tested"] = self.infected_tested(mmdd=mmdd, infectious_period=infectious_period)
        df["infected_nottested"] = self.infected_nottested(mmdd=mmdd, infectious_period=infectious_period,
                                                           undetected_multiplier=undetected_multiplier)
        df["removed_tested"] = self.removed_tested(mmdd=mmdd, infectious_period=infectious_period)
        df["removed_nottested"] = self.removed_nottested(mmdd=mmdd, infectious_period=infectious_period,
                                            undetected_multiplier=undetected_multiplier)
        # replace negative values in dataframe by 0, negatives occur due to corrections in RIVM data
        num = df._get_numeric_data()
        num[num < 0] = 0
        if return_integer:
            df = df.round(0)
        df["inhabitant"] = df["susceptible"] + df["exposed"] + df["infected_tested"] + df["infected_nottested"] + df["removed_tested"] + df["removed_nottested"]
        return df

    def concentrated_init(self, people_exposed, municipality):
        df = pd.DataFrame()
        df["susceptible"] = gemeente_shapes['inhabitant']
        df["exposed"]=0
        df["infected_tested"]=0
        df["infected_nottested"]=0
        df["removed_tested"]=0
        df["removed_nottested"]=0
        # Test if people exposed < inhabitants:
        if df.loc[municipality]['susceptible']>=people_exposed:
            df.loc[df.index == municipality, 'exposed'] += people_exposed
            df.loc[df.index == municipality, 'susceptible'] -= people_exposed
            return df
        else:
            print("Mistake: Exposed can not be larger than Inhabitants")
            return None

    def evenlydistributed_init(self, people_exposed, return_integer=False):
        import numpy as np
        total_population = np.sum(gemeente_shapes['inhabitant'])
        df = pd.DataFrame()
        df["susceptible"] = gemeente_shapes['inhabitant']-people_exposed/total_population*gemeente_shapes['inhabitant']
        df["exposed"]=people_exposed/total_population*gemeente_shapes['inhabitant']
        df["infected_tested"]=0
        df["infected_nottested"]=0
        df["removed_tested"]=0
        df["removed_nottested"]=0
        if return_integer:
            return df.round(0)
        return df

    @classmethod
    def mmdd2date(cls,mmdd,year=2020,hour=14):
        if len(mmdd)==8:
            year=int(mmdd[:4])
            mmdd = mmdd[4:]
        return datetime(day=int(mmdd[2:]),month=int(mmdd[:2]),year=year,hour=hour)
    @classmethod
    def date2mmdd(cls,date):
        if date.year==2020:
            return f'{date.month:02d}{date.day:02d}'
        return f'{date.year:04d}{date.month:02d}{date.day:02d}'

rivm = RivmLoader()